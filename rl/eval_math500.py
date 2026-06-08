#!/usr/bin/env python
"""OLD BENCHMARK, MATH-500 IS TOO HARD FOR HRM. SEE evanl_math.py INSTEAD"""
"""MATH-500 benchmark accuracy for an HRM checkpoint (greedy decode).

Reports pass@1 (greedy) on HuggingFaceH4/MATH-500 using the same cot,synth
PrefixLM prompting and last-\\boxed{} + Math-Verify scoring as DAPO. Pass
--adapter to eval a LoRA (SFT or DAPO); omit to eval the base model.
"""
import argparse, json, os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed, is_correct
from datasets import load_dataset
BOQ, EOQ, EOA, PAD = 6, 7, 11, 5; COND = {"cot": 9, "synth": 13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=50)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).cuda().eval()
    print(f"loaded {args.tag} (adapter={args.adapter or 'none'})", flush=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [{"problem": ex["problem"], "answer": ex["answer"]} for ex in ds][:args.limit]

    def pid(p):
        return [BOQ, COND["cot"], COND["synth"]] + tok(p, add_special_tokens=False)["input_ids"] + [EOQ]

    correct = boxed = 0
    for s in range(0, len(rows), args.micro):
        ch = rows[s:s + args.micro]; ps = [pid(r["problem"]) for r in ch]
        T = max(len(p) for p in ps); B = len(ch)
        inp = torch.full((B, T), PAD); attn = torch.zeros((B, T), dtype=torch.long)
        for b, p in enumerate(ps):
            inp[b, T - len(p):] = torch.tensor(p); attn[b, T - len(p):] = 1
        inp = inp.long().cuda(); attn = attn.cuda()
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=attn, token_type_ids=attn.clone(),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=PAD, eos_token_id=EOA)
        for b, r in enumerate(ch):
            g = out[b, T:]; eos = (g == EOA).nonzero()
            glen = int(eos[0]) + 1 if len(eos) else len(g)
            txt = tok.decode(g[:glen], skip_special_tokens=True)
            hb = extract_last_boxed(txt) is not None
            boxed += int(hb); correct += int(is_correct(txt, r["answer"]) if hb else False)
        print(f"  {s+B}/{len(rows)} running acc={correct/(s+B):.3f}", flush=True)
    print(f"\n==== MATH-500 [{args.tag}] ====\n"
          f"  pass@1(greedy) = {correct/len(rows):.4f}  ({correct}/{len(rows)})\n"
          f"  boxed_rate     = {boxed/len(rows):.4f}", flush=True)


if __name__ == "__main__":
    main()
