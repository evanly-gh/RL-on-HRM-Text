#!/usr/bin/env python
"""Generate cached demo outputs: base vs SFT vs DAPO on curated MATH problems.

For each problem and model, greedy-decode with the paper's synth,cot prompting,
extract the last \\boxed{}, check correctness. Writes a markdown file (for slides)
and a json. Everything is pre-generated so the live demo has zero waiting.
"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.path.dirname(__file__))
from reward import extract_last_boxed, is_correct
BOQ, EOQ, EOA, PAD = 6, 7, 11, 5; COND = {"synth": 13, "cot": 9}

# Curated demo problems (level-spread; L5s likely expose failures). gold = boxed answer.
PROBLEMS = [
    ("L1", r"What is the value of $3^2 + 4^2$?", r"\boxed{25}"),
    ("L2", r"If $f(x) = 2x + 3$, what is $f(5)$?", r"\boxed{13}"),
    ("L3", r"A bag has 4 red and 6 blue marbles. Two are drawn without replacement. What is the probability both are red? Express as a common fraction.", r"\boxed{\frac{2}{15}}"),
    ("L4", r"Find the remainder when $7^{2024}$ is divided by $100$.", r"\boxed{1}"),
    ("L5", r"Let $a,b,c$ be positive reals with $a+b+c=1$. Find the minimum of $\frac{1}{a}+\frac{1}{b}+\frac{1}{c}$.", r"\boxed{9}"),
    ("L5", r"How many ordered triples $(a,b,c)$ of positive integers satisfy $a+b+c = 10$?", r"\boxed{36}"),
]


def load(model_id, adapter):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).cuda().eval()
    return m


def gen(model, tok, problem, max_new):
    ids = [BOQ, COND["synth"], COND["cot"]] + tok(problem, add_special_tokens=False)["input_ids"] + [EOQ]
    inp = torch.tensor([ids]).cuda(); attn = torch.ones_like(inp)
    with torch.no_grad():
        out = model.generate(input_ids=inp, attention_mask=attn, token_type_ids=attn.clone(),
                             max_new_tokens=max_new, do_sample=False, pad_token_id=PAD, eos_token_id=EOA)
    g = out[0, inp.shape[1]:]; eos = (g == EOA).nonzero(); glen = int(eos[0]) + 1 if len(eos) else len(g)
    return tok.decode(g[:glen], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=""); ap.add_argument("--dapo", default="")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--problems", default="", help="jsonl with {level,problem,gold}; else use built-in")
    ap.add_argument("--out", default="/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026/runs/demo")
    args = ap.parse_args()
    global PROBLEMS
    if args.problems:
        PROBLEMS = [(r["level"], r["problem"], r["gold"])
                    for r in (json.loads(l) for l in open(args.problems) if l.strip())]
    tok = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B", trust_remote_code=True)
    models = [("base", "")]
    if args.sft: models.append(("SFT", args.sft))
    if args.dapo: models.append(("DAPO", args.dapo))

    results = {p[1]: {"level": p[0], "gold": p[2], "outputs": {}} for p in PROBLEMS}
    for name, adapter in models:
        m = load("sapientinc/HRM-Text-1B", adapter)
        for lvl, prob, gold in PROBLEMS:
            txt = gen(m, tok, prob, args.max_new_tokens)
            box = extract_last_boxed(txt)
            ok = bool(is_correct(txt, gold)) if box else False
            results[prob]["outputs"][name] = {"text": txt, "boxed": box, "correct": ok}
            print(f"[{name}] {lvl} correct={ok} boxed={box}", flush=True)
        del m; torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(f"{args.out}/demo.json", "w"), indent=2)
    with open(f"{args.out}/demo.md", "w") as f:
        for prob, d in results.items():
            f.write(f"\n## [{d['level']}] {prob}\n\n**Gold:** {d['gold']}\n\n")
            for name in [n for n, _ in models]:
                o = d["outputs"].get(name, {})
                mark = "✅" if o.get("correct") else "❌"
                f.write(f"### {name} {mark} (boxed={o.get('boxed')})\n\n```\n{o.get('text','')[:1200]}\n```\n\n")
    print(f"\nwrote {args.out}/demo.md and demo.json", flush=True)


if __name__ == "__main__":
    main()
