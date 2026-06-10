#!/usr/bin/env python
"""MATH eval for HRM checkpoints, paper-aligned.

Supports the paper's exact protocol: EleutherAI/hendrycks_math (full test, all
7 subjects), condition "synth,cot" (canonical order per model card + eval
config), greedy, last-\\boxed{} + Math-Verify. Also supports HuggingFaceH4/
MATH-500 for comparison. Ground truth = last \\boxed in the dataset solution
(hendrycks) or the answer field (math500).
"""
import argparse, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed, is_correct
from datasets import load_dataset, get_dataset_config_names
BOQ, EOQ, EOA, PAD = 6, 7, 11, 5
COND = {"direct": 8, "cot": 9, "noisy": 12, "synth": 13}


def load_math(dataset):
    rows = []
    if dataset == "hendrycks":
        for sub in get_dataset_config_names("EleutherAI/hendrycks_math"):
            for ex in load_dataset("EleutherAI/hendrycks_math", sub, split="test"):
                gt = extract_last_boxed(ex["solution"])
                if gt:
                    rows.append({"problem": ex["problem"], "answer": gt, "level": ex.get("level"),
                                 "type": ex.get("type") or sub})
    elif dataset == "math500":
        for ex in load_dataset("HuggingFaceH4/MATH-500", split="test"):
            rows.append({"problem": ex["problem"], "answer": ex["answer"], "level": ex.get("level"),
                         "type": ex.get("subject") or ex.get("type")})
    else:
        raise ValueError(dataset)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--dataset", default="hendrycks", choices=["hendrycks", "math500"])
    ap.add_argument("--condition", default="synth,cot")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--micro", type=int, default=33)
    ap.add_argument("--sample", type=int, default=0, help="0=all; else random N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--shard", type=int, default=0, help="this shard index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-json", default="", help="write {correct,total,boxed} here")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).cuda().eval()
    cond_ids = [COND[c.strip()] for c in args.condition.split(",")]
    print(f"[{args.tag}] dataset={args.dataset} condition={args.condition} max_new={args.max_new_tokens} "
          f"adapter={args.adapter or 'none'}", flush=True)

    rows = load_math(args.dataset)
    if args.sample and args.sample < len(rows):
        random.seed(args.seed); rows = random.sample(rows, args.sample)
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
        print(f"  shard {args.shard}/{args.num_shards}", flush=True)
    print(f"  {len(rows)} problems", flush=True)

    def pid(p):
        return [BOQ] + cond_ids + tok(p, add_special_tokens=False)["input_ids"] + [EOQ]

    correct = boxed = 0
    by_level = {}
    by_type = {}
    by_type_level = {}
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
            ok = bool(is_correct(txt, r["answer"])) if hb else False
            boxed += int(hb); correct += int(ok)
            lv = r.get("level") or "?"; by_level.setdefault(lv, [0, 0]); by_level[lv][0] += int(ok); by_level[lv][1] += 1
            ty = r.get("type") or "?"; by_type.setdefault(ty, [0, 0]); by_type[ty][0] += int(ok); by_type[ty][1] += 1
            tl = f"{ty}|{lv}"; by_type_level.setdefault(tl, [0, 0]); by_type_level[tl][0] += int(ok); by_type_level[tl][1] += 1
        print(f"  {s+B}/{len(rows)} acc={correct/(s+B):.3f}", flush=True)
    print(f"\n==== MATH [{args.tag}] {args.dataset}/{args.condition}/max_new={args.max_new_tokens} ====", flush=True)
    print(f"  pass@1(greedy) = {correct/len(rows):.4f}  ({correct}/{len(rows)})  boxed={boxed/len(rows):.4f}", flush=True)
    for lv in sorted(by_level):
        c, n = by_level[lv]; print(f"    {lv}: {c}/{n} = {c/max(1,n):.3f}", flush=True)
    for ty in sorted(by_type):
        c, n = by_type[ty]; print(f"    {ty}: {c}/{n} = {c/max(1,n):.3f}", flush=True)
    for tl in sorted(by_type_level):
        c, n = by_type_level[tl]; print(f"    [TL] {tl}: {c}/{n}", flush=True)
    if args.out_json:
        import json as _j
        _j.dump({"correct": correct, "total": len(rows), "boxed": boxed,
                 "by_level": {k: v for k, v in by_level.items()},
                 "by_type": {k: v for k, v in by_type.items()},
                 "by_type_level": {k: v for k, v in by_type_level.items()}}, open(args.out_json, "w"))


if __name__ == "__main__":
    main()
