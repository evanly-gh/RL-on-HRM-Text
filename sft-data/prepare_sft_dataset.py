import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
"""
Prepares HRM-Text SFT dataset from OpenR1-Math-220K.
Samples 20K examples, formats with HRM condition flags,
and pushes to HuggingFace: evanlyhf/HRM-Text-MATH-SFT
"""

import re
import random
from datasets import load_dataset, Dataset

# Config
SOURCE_DATASET = "open-r1/OpenR1-Math-220k"
HF_REPO        = "evanlyhf/HRM-Text-MATH-SFT"
SAMPLE_SIZE    = 20_000
SEED           = 42
CONDITION      = "cot,synth"

# Load
print(f"Loading {SOURCE_DATASET} ...")
ds = load_dataset(SOURCE_DATASET, split="train")
print(f"  Total rows: {len(ds)}")
print(f"  Columns: {ds.column_names}")

# Filter
def has_required_fields(ex):
    return (
        ex.get("problem") and
        ex.get("solution") and
        ex.get("answer") is not None and
        str(ex.get("answer", "")).strip() != ""
    )

print("Filtering ...")
ds_filtered = ds.filter(has_required_fields)
print(f"  After filtering: {len(ds_filtered)}")

# Sample
random.seed(SEED)
if len(ds_filtered) >= SAMPLE_SIZE:
    indices = random.sample(range(len(ds_filtered)), SAMPLE_SIZE)
    ds_sampled = ds_filtered.select(indices)
else:
    print(f"  WARNING: Only {len(ds_filtered)} examples available, using all.")
    ds_sampled = ds_filtered
print(f"  Sampled: {len(ds_sampled)}")

# Format
def format_example(ex):
    problem   = ex["problem"].strip()
    solution  = ex["solution"].strip()
    answer    = str(ex["answer"]).strip()
    response  = solution + f"\n\\boxed{{{answer}}}"
    return {
        "condition":   CONDITION,
        "instruction": problem,
        "response":    response,
    }

print("Formatting ...")
ds_formatted = ds_sampled.map(format_example,
                               remove_columns=ds_sampled.column_names)
print(f"  Columns: {ds_formatted.column_names}")

# Sanity check
missing = [i for i, ex in enumerate(ds_formatted)
           if not re.search(r'\\boxed\{', ex["response"])]
assert len(missing) == 0, (
    f"CRITICAL: {len(missing)} examples missing boxed answer."
)
print(f"  OK: 100% boxed coverage ({len(ds_formatted)} examples)")

# Preview
ex = ds_formatted[0]
print("\nPreview example 0:")
print(f"  condition:   {ex['condition']}")
print(f"  instruction: {ex['instruction'][:100]}...")
print(f"  response end: ...{ex['response'][-150:]}")

# Push
print(f"\nPushing to {HF_REPO} ...")
ds_formatted.push_to_hub(HF_REPO, private=False)
print(f"Done! https://huggingface.co/datasets/{HF_REPO}")
