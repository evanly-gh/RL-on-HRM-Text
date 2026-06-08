import time, torch, json, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed, is_correct
PROJ = "/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026"
BOQ, EOQ, EOA, PAD = 6, 7, 11, 5; COND = {"cot": 9, "synth": 13}
tok = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B", trust_remote_code=True); tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
model = PeftModel.from_pretrained(base, f"{PROJ}/runs/sft-openr1-v1/final_adapter").cuda().eval()
print("model ready", flush=True)
rows = [json.loads(l) for l in open("/mmfs1/home/evanly/RL-on-HRM-Text/rl-data/dapo_prompts.jsonl")][:8]
def pids(r): return [BOQ, COND["cot"], COND["synth"]] + tok(r["instruction"], add_special_tokens=False)["input_ids"] + [EOQ]
for MNT in [512, 1024, 2048]:
    items = [(r, pids(r)) for r in rows for _ in range(4)]
    T = max(len(p) for _, p in items); B = len(items)
    inp = torch.full((B, T), PAD); attn = torch.zeros((B, T), dtype=torch.long)
    for b, (_, p) in enumerate(items):
        inp[b, T - len(p):] = torch.tensor(p); attn[b, T - len(p):] = 1
    inp = inp.long().cuda(); attn = attn.cuda(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=inp, attention_mask=attn, token_type_ids=attn.clone(),
                             max_new_tokens=MNT, do_sample=True, temperature=1.0, pad_token_id=PAD, eos_token_id=EOA)
    torch.cuda.synchronize(); dt = time.time() - t
    gen = out[:, T:]
    texts = [tok.decode(g, skip_special_tokens=True) for g in gen]
    box = sum(extract_last_boxed(x) is not None for x in texts)
    cor = sum(is_correct(texts[i], items[i][0]["answer"]) for i in range(B))
    eos = sum(int((g == EOA).any()) for g in gen)
    actual = sum((g != PAD).sum().item() for g in gen)
    print(f"[bench] {B} rollouts max_new={MNT}: {dt:.1f}s  ~{actual/dt:.0f} tok/s  "
          f"finished(EOA)={eos}/{B} boxed={box}/{B} correct={cor}/{B} "
          f"peakmem={torch.cuda.max_memory_allocated()/1e9:.0f}GB", flush=True)
print("done", flush=True)
