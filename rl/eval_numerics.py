#!/usr/bin/env python
"""Numerical-sensitivity probe for the base HF model's MATH greedy accuracy.

Tests whether dtype/attention implementation materially change greedy MATH
accuracy on a FIXED problem set. If they do, it shows HRM's recurrent greedy
decode is numerically fragile -> the HF-integration-vs-native-engine difference
plausibly explains base(0.635) vs paper(0.562). Also dumps (gold,pred) pairs
from the default config for offline math_verify-version re-scoring.
"""
import json, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed, is_correct
from datasets import load_dataset, get_dataset_config_names
BOQ, EOQ, EOA, PAD = 6, 7, 11, 5; COND = {"synth": 13, "cot": 9}
N = int(os.environ.get("N", "160")); MNT = int(os.environ.get("MNT", "1536"))
MICRO = int(os.environ.get("MICRO", "40")); DUMP = os.environ.get("DUMP", "")

tok = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B", trust_remote_code=True); tok.padding_side = "left"
rows = []
for sub in get_dataset_config_names("EleutherAI/hendrycks_math"):
    for ex in load_dataset("EleutherAI/hendrycks_math", sub, split="test"):
        gt = extract_last_boxed(ex["solution"])
        if gt: rows.append({"problem": ex["problem"], "gold": gt, "level": ex.get("level")})
random.seed(0); rows = random.sample(rows, N)
def pid(p): return [BOQ, COND["synth"], COND["cot"]] + tok(p, add_special_tokens=False)["input_ids"] + [EOQ]
prompt_ids = [pid(r["problem"]) for r in rows]

def run(dtype, attn):
    model = AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B", dtype=dtype,
              trust_remote_code=True, attn_implementation=attn).cuda().eval()
    preds = []
    for s in range(0, len(rows), MICRO):
        ch = prompt_ids[s:s + MICRO]; T = max(len(p) for p in ch); B = len(ch)
        inp = torch.full((B, T), PAD); attn_m = torch.zeros((B, T), dtype=torch.long)
        for b, p in enumerate(ch):
            inp[b, T - len(p):] = torch.tensor(p); attn_m[b, T - len(p):] = 1
        inp = inp.long().cuda(); attn_m = attn_m.cuda()
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=attn_m, token_type_ids=attn_m.clone(),
                                 max_new_tokens=MNT, do_sample=False, pad_token_id=PAD, eos_token_id=EOA)
        for b in range(B):
            g = out[b, T:]; eos = (g == EOA).nonzero(); glen = int(eos[0]) + 1 if len(eos) else len(g)
            preds.append(tok.decode(g[:glen], skip_special_tokens=True))
    del model; torch.cuda.empty_cache()
    return preds

configs = [("bf16", torch.bfloat16, "sdpa"), ("fp32", torch.float32, "eager"), ("bf16", torch.bfloat16, "eager")]
results = {}
predmap = {}
for tag, dt, attn in configs:
    name = f"{tag}/{attn}"
    try:
        preds = run(dt, attn)
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True); continue
    acc = sum(is_correct(preds[i], rows[i]["gold"]) for i in range(len(rows))) / len(rows)
    boxed = sum(extract_last_boxed(p) is not None for p in preds) / len(rows)
    results[name] = acc; predmap[name] = preds
    print(f"{name}: acc={acc:.4f} boxed={boxed:.4f}", flush=True)

# answer-agreement between configs (how often greedy diverges)
names = list(predmap)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = predmap[names[i]], predmap[names[j]]
        same = sum(extract_last_boxed(a[k]) == extract_last_boxed(b[k]) for k in range(len(a)))
        print(f"  boxed-answer agreement {names[i]} vs {names[j]}: {same}/{len(a)} = {same/len(a):.3f}", flush=True)

if DUMP and predmap:
    base = predmap.get("bf16/sdpa") or next(iter(predmap.values()))
    with open(DUMP, "w") as f:
        for i, r in enumerate(rows):
            pred = extract_last_boxed(base[i])
            f.write(json.dumps({"gold": r["gold"], "pred": pred, "level": r["level"]}) + "\n")
    print(f"dumped {len(rows)} (gold,pred) pairs to {DUMP}", flush=True)
print("DONE", flush=True)
