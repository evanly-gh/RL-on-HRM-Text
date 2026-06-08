import time, json, random, sys, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed, is_correct
PROJ="/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026"
BOQ,EOQ,EOA,PAD=6,7,11,5; COND={"direct":8,"cot":9,"noisy":12,"synth":13}
NP=int(os.environ.get("NPROMPTS","16")); G=int(os.environ.get("GROUP","8"))
MNT=int(os.environ.get("MNT","1024")); MICRO=int(os.environ.get("MICRO","96"))

tok=AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B",trust_remote_code=True); tok.padding_side="left"
base=AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B",dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
model=PeftModel.from_pretrained(base,f"{PROJ}/runs/sft-openr1-v1/final_adapter").cuda().eval()
print("model ready",flush=True)
def pids(r):
    c=[COND[x.strip()] for x in r["condition"].split(",")]
    return [BOQ]+c+tok(r["instruction"],add_special_tokens=False)["input_ids"]+[EOQ]

def evalfile(path,seed=0):
    random.seed(seed)
    rows=[json.loads(l) for l in open(path,encoding="utf-8")]
    random.shuffle(rows); rows=rows[:NP]
    items=[(ri,pids(r)) for ri,r in enumerate(rows) for _ in range(G)]
    box=cor=trunc=0; per=[[] for _ in rows]; t0=time.time(); ntok=0
    for s in range(0,len(items),MICRO):
        ch=items[s:s+MICRO]; T=max(len(p) for _,p in ch); B=len(ch)
        inp=torch.full((B,T),PAD); attn=torch.zeros((B,T),dtype=torch.long)
        for b,(_,p) in enumerate(ch):
            inp[b,T-len(p):]=torch.tensor(p); attn[b,T-len(p):]=1
        inp=inp.long().cuda(); attn=attn.cuda()
        with torch.no_grad():
            out=model.generate(input_ids=inp,attention_mask=attn,token_type_ids=attn.clone(),
                max_new_tokens=MNT,do_sample=True,temperature=1.0,pad_token_id=PAD,eos_token_id=EOA)
        gen=out[:,T:]
        for b,(ri,_) in enumerate(ch):
            g=gen[b]; eos=(g==EOA).nonzero()
            glen=int(eos[0])+1 if len(eos) else len(g); ntok+=glen
            txt=tok.decode(g[:glen],skip_special_tokens=True)
            lb=extract_last_boxed(txt); hb=lb is not None
            box+=int(hb)
            if not hb and len(eos)==0: trunc+=1
            ok=is_correct(txt,rows[ri]["answer"]) if hb else False
            cor+=int(ok); per[ri].append(ok)
    dt=time.time()-t0; N=len(items)
    passG=np.mean([any(c) for c in per]); nondeg=np.mean([0<sum(c)<len(c) for c in per])
    print(f"\n=== {os.path.basename(path)} | {NP} prompts x{G} = {N} rollouts, max_new={MNT} ===")
    print(f"  time={dt:.0f}s  ~{ntok/dt:.0f} tok/s   boxed={box/N:.2f} trunc={trunc/N:.2f} "
          f"solve={cor/N:.3f} pass@{G}={passG:.2f} nondegen_groups={nondeg:.2f}",flush=True)
    return nondeg

for path in ["rl-data/dapo_prompts.jsonl","rl-data/openr1_dapo_prompts.jsonl"]:
    evalfile("/mmfs1/home/evanly/RL-on-HRM-Text/"+path)
print("\nDONE",flush=True)
