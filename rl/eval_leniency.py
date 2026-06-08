#!/usr/bin/env python
"""Quantify how much of base MATH accuracy depends on math_verify's symbolic
equivalence (vs exact normalized string match), and dump examples to eyeball
for false positives. Helps explain base(0.635) >> paper(0.565).
"""
import argparse, os, sys, re, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed
from math_verify import parse, verify
from datasets import load_dataset, get_dataset_config_names
BOQ, EOQ, EOA, PAD = 6, 7, 11, 5; COND = {"synth": 13, "cot": 9}


def inner(boxed):  # strip \boxed{...}
    return boxed[len(r"\boxed{"):-1] if boxed and boxed.startswith(r"\boxed{") else boxed


def norm(s):
    if s is None: return None
    s = s.strip()
    for a, b in [(r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\ ", ""),
                 (" ", ""), ("$", ""), (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"),
                 ("\\\n", ""), ("\n", "")]:
        s = s.replace(a, b)
    return s.rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--micro", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026/runs/leniency_dump.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B", trust_remote_code=True); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    rows = []
    for sub in get_dataset_config_names("EleutherAI/hendrycks_math"):
        for ex in load_dataset("EleutherAI/hendrycks_math", sub, split="test"):
            gt = extract_last_boxed(ex["solution"])
            if gt: rows.append({"problem": ex["problem"], "gold": gt, "level": ex.get("level")})
    random.seed(args.seed); rows = random.sample(rows, args.n)

    def pid(p): return [BOQ, COND["synth"], COND["cot"]] + tok(p, add_special_tokens=False)["input_ids"] + [EOQ]

    mv_correct = strict_correct = both = mv_only = 0
    dump = open(args.dump, "w")
    for s in range(0, len(rows), args.micro):
        ch = rows[s:s + args.micro]; ps = [pid(r["problem"]) for r in ch]
        T = max(len(p) for p in ps); B = len(ch)
        inp = torch.full((B, T), PAD); attn = torch.zeros((B, T), dtype=torch.long)
        for b, p in enumerate(ps):
            inp[b, T - len(p):] = torch.tensor(p); attn[b, T - len(p):] = 1
        inp = inp.long().cuda(); attn = attn.cuda()
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=attn, token_type_ids=attn.clone(),
                                 max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=PAD, eos_token_id=EOA)
        for b, r in enumerate(ch):
            g = out[b, T:]; eos = (g == EOA).nonzero(); glen = int(eos[0]) + 1 if len(eos) else len(g)
            txt = tok.decode(g[:glen], skip_special_tokens=True)
            pred = extract_last_boxed(txt)
            mv = False
            if pred is not None:
                try: mv = bool(verify(parse(r["gold"]), parse(pred)))
                except Exception: mv = False
            strict = (pred is not None) and (norm(inner(pred)) == norm(inner(r["gold"])))
            mv_correct += mv; strict_correct += strict
            if mv and strict: both += 1
            if mv and not strict:
                mv_only += 1
                dump.write(json.dumps({"level": r["level"], "gold": inner(r["gold"]),
                                       "pred": inner(pred) if pred else None}) + "\n")
    dump.close()
    N = len(rows)
    print(f"\n==== LENIENCY (base, {N} hendrycks, 3072) ====", flush=True)
    print(f"  math_verify acc = {mv_correct/N:.4f}", flush=True)
    print(f"  strict-string acc = {strict_correct/N:.4f}", flush=True)
    print(f"  credited by math_verify but NOT strict-equal = {mv_only} ({mv_only/N:.3f} of all) -> equivalence credit", flush=True)
    print(f"  (dumped those {mv_only} gold/pred pairs to {args.dump} for manual false-positive check)", flush=True)


if __name__ == "__main__":
    main()
