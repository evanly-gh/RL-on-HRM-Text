#!/usr/bin/env python
"""Build the DAPO prompt pool (problem + verifiable ground-truth answer).

Source: BytedTsinghua-SIA/DAPO-Math-17k. Its `reward_model.ground_truth` is a
clean closed-form/integer answer suitable for Math-Verify. We strip DAPO-17k's
own instruction framing and re-wrap each problem in HRM's {condition,instruction}
format. The gold `answer` is kept so the online reward function can verify
rollouts. Output: rl-data/dapo_prompts.jsonl
    {"condition": "cot,synth", "instruction": "<bare problem>", "answer": "<gt>"}
"""
import argparse, json, re
from datasets import load_dataset

HEADER = re.compile(r'^Solve the following math problem.*?problem\.\s*', re.DOTALL)
FOOTER = 'Remember to put your answer on its own line after "Answer:".'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rl-data/dapo_prompts.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    n, dup = 0, 0
    seen = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            content = ex["prompt"][0]["content"]
            problem = HEADER.sub("", content).replace(FOOTER, "").strip()
            gt = ex["reward_model"]["ground_truth"]
            if not problem or gt is None or str(gt).strip() == "":
                continue
            key = problem[:200]
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            f.write(json.dumps({"condition": "cot,synth", "instruction": problem,
                                "answer": str(gt).strip()}, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} DAPO prompts to {args.out} ({dup} duplicates skipped)")


if __name__ == "__main__":
    main()
