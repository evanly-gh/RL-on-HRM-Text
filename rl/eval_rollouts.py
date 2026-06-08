#!/usr/bin/env python
"""Measure SFT-model rollout health on a DAPO prompt sample.

Exercises the exact generation path DAPO uses (HF generate + token_type_ids=1
on the prompt) and reports:
  boxed_rate      fraction of rollouts containing a \\boxed{}
  truncation_rate fraction hitting max_new_tokens with no \\boxed{}
  solve_rate      fraction correct (Math-Verify vs gold)
  pass@G          fraction of prompts with >=1 correct rollout in its group
  nondegen_rate   fraction of prompt-groups with mixed correct/wrong (DAPO signal)
  mean_gen_len    avg generated tokens

Use this to (a) confirm SFT produced \\boxed{} reliably and (b) check the prompt
pool is the right difficulty (nondegen_rate well above 0).
"""
import argparse, json, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from reward import extract_last_boxed, is_correct

BOQ, EOQ, EOA, PAD = 6, 7, 11, 5
COND = {"direct": 8, "cot": 9, "noisy": 12, "synth": 13}


def build_prompt_ids(tok, condition, instruction):
    cond_ids = [COND[c.strip()] for c in condition.split(",")]
    inst = tok(instruction, add_special_tokens=False)["input_ids"]
    return [BOQ] + cond_ids + inst + [EOQ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--adapter", required=True, help="SFT LoRA adapter dir")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--micro", type=int, default=64, help="sequences per generate call")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = PAD
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                trust_remote_code=True).cuda().eval()
    model = PeftModel.from_pretrained(base, args.adapter).cuda().eval()

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    random.shuffle(rows)
    rows = rows[:args.n_prompts]

    # expand to (prompt, group) rollouts
    items = []  # (row_idx, prompt_ids)
    for ri, r in enumerate(rows):
        pid = build_prompt_ids(tok, r["condition"], r["instruction"])
        for _ in range(args.group):
            items.append((ri, pid))

    boxed = trunc = correct = 0
    gen_lens = []
    per_prompt_correct = [[] for _ in rows]

    for s in range(0, len(items), args.micro):
        chunk = items[s:s + args.micro]
        maxlen = max(len(p) for _, p in chunk)
        B = len(chunk)
        input_ids = torch.full((B, maxlen), PAD, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for b, (_, p) in enumerate(chunk):
            input_ids[b, maxlen - len(p):] = torch.tensor(p)  # left pad
            attn[b, maxlen - len(p):] = 1
        input_ids = input_ids.cuda(); attn = attn.cuda()
        ttids = attn.clone()  # prompt tokens (non-pad) are prefix => type 1
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn, token_type_ids=ttids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                pad_token_id=PAD, eos_token_id=EOA)
        gen = out[:, maxlen:]
        for b, (ri, _) in enumerate(chunk):
            g = gen[b]
            # length up to EOA
            eos_pos = (g == EOA).nonzero()
            glen = int(eos_pos[0]) + 1 if len(eos_pos) else len(g)
            gen_lens.append(glen)
            text = tok.decode(g[:glen], skip_special_tokens=True)
            lb = extract_last_boxed(text)
            has_box = lb is not None
            boxed += int(has_box)
            hit_cap = (len(eos_pos) == 0)
            if hit_cap and not has_box:
                trunc += 1
            ok = is_correct(text, rows[ri]["answer"]) if has_box else False
            correct += int(ok)
            per_prompt_correct[ri].append(ok)
        print(f"  rollouts {s+len(chunk)}/{len(items)} ...", flush=True)

    N = len(items)
    passG = np.mean([any(c) for c in per_prompt_correct])
    nondeg = np.mean([0 < sum(c) < len(c) for c in per_prompt_correct])
    print("\n==== ROLLOUT HEALTH ====")
    print(f"prompts={len(rows)} group={args.group} rollouts={N} max_new={args.max_new_tokens}")
    print(f"boxed_rate      = {boxed/N:.3f}")
    print(f"truncation_rate = {trunc/N:.3f}")
    print(f"solve_rate      = {correct/N:.3f}")
    print(f"pass@{args.group}         = {passG:.3f}")
    print(f"nondegen_rate   = {nondeg:.3f}   (groups with mixed correct/wrong)")
    print(f"mean_gen_len    = {np.mean(gen_lens):.0f}  (max {max(gen_lens)})")


if __name__ == "__main__":
    main()
