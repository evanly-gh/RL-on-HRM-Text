# HRM-Text-1B Post-Training: Complete Research & Implementation Plan

**Goal:** Hill-climb MATH benchmark from 56.2% using SFT + RL on HRM-Text-1B  
**Timeline:** One week  
**Hardware:** Hyak Klone — partition `gpu-h200` (H200, preemptible ckpt)  
**Research basis:** 9 rounds of adversarial deep-research (~900 agents, ~20M tokens verified)

---

## Part 0: The Three Problems From the Prior Experiment (Explained Plainly)

Before anything else — here is the complete story of what went wrong before and why every fix in this document exists.

### Problem 1: `\boxed{}` — A Chain of Three Failures

The DAPO RL stage received reward=0 for every single rollout. Nothing trained. Here's why:

**Failure A — Wrong dataset format.** DAPO-Math-17K was used as SFT training data. That dataset uses `"Answer: 42"` (AIME competition integer format), NOT `\boxed{42}`. After SFT, the model learned to end every solution with `"Answer: N"`. During DAPO rollout evaluation, Math-Verify searched for `\boxed{}` to extract the predicted answer. Found none. Returned reward=0 for every rollout. Training ran for hundreds of steps, the model received no gradient signal, nothing improved.

**Failure B — Token cap too short.** Generation was capped at 2048 tokens, but HRM's pretraining context is 4096 tokens. MATH level 4-5 solutions regularly exceed 2000 tokens. The model started writing solutions, hit the wall mid-sentence, stopped — it never reached the point where `\boxed{}` appears. Every truncated response got reward=0.

**Failure C — Intermediate `\boxed{}` values in OpenR1.** When we switched to OpenR1-Math-220K (which does use `\boxed{}`), there was a subtler problem. DeepSeek-R1 (which generated those solutions) frequently self-corrects mid-reasoning:

> *"...so the answer is `\boxed{5}`. Wait, I made an arithmetic error. Let me recalculate... the answer is `\boxed{7}`."*

A bug in Math-Verify concatenated ALL `\boxed{}` occurrences into a set `{5, 7}` and tried to match that against the ground truth — which never works. This caused ~10% MATH benchmark score degradation. The fix: **always extract only the LAST `\boxed{}`**, which is the final answer after any self-correction.

**All three fixes are now in this plan:**
1. Primary SFT dataset changed to OpenR1-Math-220K + MATH train (both have `\boxed{}`)
2. Token cap raised to 4096 (HRM's full pretraining context)
3. Reward function extracts last `\boxed{}`, not first or all
4. Data pipeline asserts 100% `\boxed{}` coverage before any training starts

---

### Problem 2: Why TRL SFTTrainer Cannot Be Used

**What TRL is:** HuggingFace's `SFTTrainer` is a general-purpose supervised fine-tuning library built for standard GPT-style causal language models.

**What HRM needs — PrefixLM attention:** HRM was pretrained with a hybrid attention pattern. When HRM processes a `{instruction, response}` pair:
- Instruction tokens see **each other bidirectionally** — like a BERT encoder. The model reads the entire problem before starting to generate.
- Response tokens attend **causally** — like a GPT decoder. Each solution token only sees prior tokens.

This bidirectional-then-causal pattern is implemented in HRM's custom FlashAttention kernel (`flash_attention_prefixlm_v2.py`), which takes `prefix_lens` (how many instruction tokens per packed sequence) and `causal_lens` (how many response tokens) to construct the correct attention mask.

At inference time via the HuggingFace API, this same attention pattern is controlled by `token_type_ids`: setting `token_type_ids=1` on the instruction positions tells the model to apply bidirectional attention there. The HF model card is explicit: *"If you omit token_type_ids, attention falls back to pure causal, which does not match the pre-training distribution and will give noticeably worse logits."*

**What TRL does wrong:** TRL SFTTrainer is designed for causal LMs. It has no concept of `prefix_lens`/`causal_lens`. When it trains HRM, it applies causal attention to ALL tokens including the instruction. This means every gradient update during SFT is computed under the wrong attention pattern — one the model was never pretrained with. The result: the model trains itself to work in a different, incorrect mode, potentially corrupting the representations it spent 40B tokens building.

**What to use instead:** HRM's own `pretrain.py` with `cfg_sft` config, OR the HuggingFace version of HRM with a custom training loop that passes `token_type_ids` correctly. PEFT LoRA works fine on top of this because LoRA only modifies weight matrices — it does not touch the attention mask logic, which the model handles entirely on its own via `token_type_ids`.

---

### Problem 3: Why vLLM Cannot Be Used for DAPO Rollouts

**What vLLM is:** A fast inference engine for LLMs. For DAPO we generate 8 responses per math problem, potentially for thousands of problems across hundreds of training steps. vLLM's PagedAttention is typically 5-10× faster than HuggingFace's `model.generate()`.

**What breaks:** vLLM's PagedAttention memory management assumes standard causal attention. There is no mechanism in PagedAttention to apply bidirectional attention to prefix tokens during generation. Running HRM through vLLM means the instruction tokens get causal attention — the wrong pattern, different from pretraining.

The HRM model card explicitly states: vLLM support is **"currently in progress"** as of the model's release. It literally doesn't work yet.

**Why this matters for DAPO:** The rollout responses generated by vLLM would come from a model operating in the wrong attention mode. Those wrong-attention responses then get fed back as training data during the DAPO update. You'd be training the model to optimize from a subtly corrupted distribution of its own behavior.

**What to use instead:** HuggingFace `model.generate()` with `token_type_ids=torch.ones_like(input_ids)`. Slower, but correct. For a 1B model on an H200, this is fast enough.

---

## Part 0.5: Handoff Reference — What A New Implementation Chat Needs

This section is a self-contained quick-start for implementation. Everything needed to write the actual training scripts is here.

### Exact Model Loading (from Official Model Card)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "sapientinc/HRM-Text-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,   # REQUIRED — HRM uses custom architecture
).cuda().eval()
```

### Exact Condition Token Mapping

The `condition` field in the JSONL data maps to specific special tokens that get prepended to the prompt:

| Condition string | Token(s) | Format in prompt |
|-----------------|----------|-----------------|
| `"cot,synth"` | `<\|quad_end\|><\|object_ref_end\|>` | `<\|im_start\|><\|quad_end\|><\|object_ref_end\|>[instruction]<\|im_end\|>` |
| `"direct"` | `<\|object_ref_start\|>` | `<\|im_start\|><\|object_ref_start\|>[instruction]<\|im_end\|>` |
| `"synth"` first, `"cot"` second — order matters | Combined as shown | Flags prepended left-to-right |

**How to build prompts for training/inference:**
```python
def format_hrm_prompt(condition: str, instruction: str) -> str:
    """
    Build an HRM-formatted prompt string.
    The tokenizer handles special token mapping when condition tokens
    are included inside the im_start/im_end envelope.
    """
    # The data_io pipeline handles this formatting internally.
    # For the HF model at inference/training, format as:
    return f"<|im_start|>{condition}{instruction}<|im_end|>"
    # Where condition is the raw string "cot,synth" — the tokenizer
    # maps this to the special token sequence automatically.
```

### Exact Generation Code (from Official Model Card)

```python
# The official example from the HRM-Text-1B model card:
condition = "<|quad_end|><|object_ref_end|>"  # = "synth,cot" mapping
prompt = f"<|im_start|>{condition}Explain why the sky is blue.<|im_end|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# CRITICAL: token_type_ids MUST be set. Without it: pure causal attention → wrong outputs.
inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=False))
```

**For math SFT generation (with sampling):**
```python
inputs = tokenizer(math_prompt, return_tensors="pt").to(model.device)
inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])  # always set this

outputs = model.generate(
    **inputs,
    max_new_tokens=4096,   # HRM context_size=4096; don't go lower for math
    do_sample=True,
    temperature=0.9,
    num_return_sequences=8,   # for DAPO group_size=8
    pad_token_id=tokenizer.eos_token_id,
)
```

### Forward Pass for Training

```python
# token_type_ids shape: [batch_size, seq_len] — same as input_ids
# 1 = instruction token (bidirectional attention)
# 0 = response token (causal attention)

outputs = model(
    input_ids=input_ids,              # [B, T]
    token_type_ids=token_type_ids,    # [B, T] — MUST be passed
    attention_mask=attention_mask,    # [B, T] — standard padding mask
    labels=labels,                    # [B, T] — -100 for instruction tokens
)
loss = outputs.loss
```

### Environment Setup

```bash
# 1. Clone the repo
git clone https://github.com/sapientinc/HRM-Text
cd HRM-Text

# 2. Install FlashAttention 3 (REQUIRED for H100/H200 — not pip install flash-attn)
pip install ninja
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=4 python setup.py install
cd ../..

# 3. Install repo dependencies
pip install -r requirements.txt
# Key packages: transformers>=5.9.0, peft, torch>=2.2, hydra-core, omegaconf

# 4. Verify
python -c "
from transformers import AutoModelForCausalLM
from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm
print('Environment OK')
"
```

### Data Format Summary

Every training example, both SFT and DAPO prompts:

```json
{"condition": "cot,synth", "instruction": "[math problem text]", "response": "[step-by-step solution ending with \\boxed{answer}]"}
```

For DAPO prompts (no response field):
```json
{"condition": "cot,synth", "instruction": "[math problem text]"}
```

### Key Numbers

| Parameter | Value | Why |
|-----------|-------|-----|
| context_size | 4096 | HRM pretraining context; max_seq_length and max_new_tokens ceiling |
| SFT dataset size | ~27.5K examples | OpenR1 filtered 20K + MATH train 7.5K |
| SFT epochs | 1 | Over-SFT degrades downstream RL |
| LoRA rank | 16 | ~0.5% of 1B params trainable |
| LoRA alpha | 16 | alpha=rank is standard |
| LoRA dropout | 0.05 | Small regularization |
| DAPO ε_high | 0.28 | Delays entropy collapse |
| DAPO ε_low | 0.20 | Same as vanilla GRPO |
| DAPO β_kl | 0.0 | No KL penalty for math RL |
| DAPO group_size | 8 | Rollouts per prompt |
| DAPO max_new_tokens | 4096 | Hard context limit |
| Reference reset | Every 250 steps | Prevents KL drift |
| DAPO condition flag | cot,synth | CoT rollouts throughout |
| Gradient clip | 1.0 | Recurrent BPTT amplifies gradients |
| Learning rate (SFT) | 2e-5 | Standard for LoRA SFT |
| LR scheduler | cosine | Standard |

### Files You Need to Write

The HRM-Text repo does NOT include RL training scripts. You need to create:

1. **`sft/train_hrm_sft.py`** — Custom SFT loop using HF model + PEFT LoRA + token_type_ids
2. **`rl/train_dapo.py`** — DAPO training loop using HF generate() for rollouts
3. **`rl/train_reinforce_pp.py`** — REINFORCE++ variant with global batch normalization
4. **`data/prepare_data.py`** — Runs the OpenR1 filtering + format conversion pipeline

The repo DOES include (use these):
- `pretrain.py` — native training loop (Path B for SFT without LoRA)
- `scripts/prepare_sft_data.py` — tokenizes data for native stack
- `config/cfg_sft.yaml` — SFT config for native stack
- `models/flash_attention_prefixlm_v2.py` — PrefixLM kernel (don't modify)
- `conversion/convert_to_hf.py` — export to HuggingFace format

---

## Part 1: HRM-Text-1B Architecture — What Makes It Different

### The Hierarchical Recurrent Design

HRM-Text-1B is not a standard transformer. Standard transformers read input once and produce output. HRM runs a nested **recurrent computation loop** before producing each output token — the model "deliberates" across multiple passes before deciding what to write.

```
For each output token:
  ┌──────────────────────────────────────────────────────┐
  │  H-cycle 1:                                          │
  │    L-step 1: L-module processes input → updates z_L  │
  │    L-step 2: L-module runs again on updated state    │
  │    L-step 3: L-module runs again                     │
  │    H-step:   H-module processes z_L → updates z_H   │
  │              State injection: z_L = z_L + z_H        │
  │                                                      │
  │  H-cycle 2: (same structure with updated state)      │
  └──────────────────────────────────────────────────────┘
  → LM Head produces logits for next token
```

**H-module (High-level / Slow):** Maintains strategic context. Updates less frequently. Handles high-level reasoning direction.

**L-module (Low-level / Fast):** Handles execution detail. Runs multiple steps per H update. Does the fine-grained computation.

**Weight-tying:** The same L-module weights are used across all 3 L-steps. The same H-module weights are used across both H-cycles. This keeps the parameter count at 1B while getting 8 total computation passes per token — effectively "depth for free." The tradeoff is that LoRA applied to these modules gets called N times per forward pass.

**Why this achieves 56.2% MATH at 1B parameters:** More computation per parameter at inference time. Standard 1B transformers have one pass through their layers. HRM gets 8 equivalent "layers" worth of processing from the same parameter count, by reusing weights across the recurrent loop.

### PrefixLM Attention — Why It Matters for Post-Training

HRM was pretrained on instruction-response pairs, not raw text. Every training example was structured:

```
[INSTRUCTION: math problem text] → [RESPONSE: step-by-step solution with \boxed{}]
```

During pretraining, the instruction tokens attended bidirectionally (saw the full problem), and the response tokens attended causally (generated one token at a time). The model learned to *read the whole problem before writing the solution*.

This is why `token_type_ids` matters so much: it tells the model which tokens are "instruction" (bidirectional) and which are "response" (causal). Get this wrong during SFT or DAPO, and you're training a fundamentally different model than what was pretrained.

### Condition Flags — What They Are and Why `cot,synth`

Condition flags are **special tokens prepended to the instruction field** that were present in all pretraining data. They attend bidirectionally as part of the prefix and condition the model's output style. Two flags are combined for our use:

- **`cot`** (chain-of-thought): tells the model to produce step-by-step reasoning. Without this, the model may produce direct answers which break the DAPO reward signal (no reasoning steps, potentially no `\boxed{}`).
- **`synth`** (synthetic): tells the model this is a generated/formatted response, not raw web text. Our SFT data is generated by DeepSeek-R1, and our DAPO rollouts are self-generated — `synth` is the correct flag.

**Why not `direct`:** The `direct` flag tells the model to give a short direct answer with no reasoning steps. DAPO is designed for long-CoT reasoning — the DAPO paper uses max_new_tokens=20,480 specifically because reasoning chains need to be long. More importantly, if SFT was done with `cot,synth` and DAPO uses `direct,synth`, there's a distribution shift: the model trained to write reasoning steps gets prompted to write direct answers, which it may do poorly and inconsistently.

**Why not `noisy`:** This flag is for web-crawled irregular text. Our data is structured and clean. Using it would degrade output quality.

### Training Objective

HRM was pretrained with **task-completion NLL** — the loss is computed only on response tokens, not instruction tokens. This means the model was never trained to predict the problem text — only to produce the solution given the problem. Your SFT data pipeline must replicate this: set `labels=-100` for instruction tokens, compute loss only on response tokens.

### Pretraining Configuration (Relevant for Post-Training Decisions)

| Setting | Value | Why It Matters |
|---------|-------|---------------|
| Context size | 4096 tokens | Sets max_seq_length and max_new_tokens ceiling |
| Global batch | 172,032 tokens/step | Token-count batching (not sample-count) |
| BPTT steps | Up to 5 (warmup from 2) | Full 5-step BPTT during fine-tuning is safe |
| FlashAttention | FA3 (Hopper GPUs only) | Requires H100/H200; A100 needs FA2 fallback check |
| Distributed training | FSDP2 for pretraining | Not needed for single-GPU fine-tuning |

### What "Latent Reasoning" Means vs Explicit CoT

HRM's paper mentions stripping `<think>` tokens before training. This means the model's internal deliberation — the H/L recurrent computation — happens in hidden state space, not in output text. The model "thinks" internally as it runs through H-cycles and L-steps.

**But this does NOT mean the response field should be empty or short.** The response field is what gets trained via NLL. Writing step-by-step math solutions in the response field is exactly correct — the model produces explicit reasoning text that is also supervised. The internal latent computation and the explicit output text are two orthogonal things. `cot,synth` teaches the model to produce structured, explicit reasoning in the output, which is what DAPO needs to evaluate rewards.

---

## Part 2: HRM-Text Repo Structure (For Implementation)

The HRM-Text repo was primarily designed for pretraining, but it DOES have native SFT support. Key confirmed facts (3-0 from README analysis):

```
sapientinc/HRM-Text/
├── pretrain.py              ← Main training loop for BOTH pretraining AND SFT
├── scripts/
│   └── prepare_sft_data.py ← Pre-tokenizes JSONL data for SFT; takes --train --tokenizer --output --epochs
├── config/
│   ├── cfg_pretrain.yaml   ← Pretraining config (Hydra)
│   └── cfg_sft.yaml        ← SFT config — use this for fine-tuning
├── dataset_new.py           ← Data loader; handles prefix_lens/causal_lens/cu_seqlens packing
├── multipack_sampler.py     ← Bin-packing sampler for token-count batching
├── models/
│   ├── baselines/hrm_nocarry_bp_warmup.py  ← Core HRM architecture
│   ├── flash_attention_prefixlm_v2.py      ← Custom PrefixLM FA kernel
│   │     Signature: flash_attn_varlen_prefixlm(q,k,v,is_causal,
│   │       prefix_lens, causal_lens, cu_seqlens, ...)
│   └── layers.py            ← Attention + MLP implementations
└── conversion/
    └── convert_to_hf.py     ← Export FSDP2 checkpoint → HuggingFace format
```

### Native SFT Launch Command (from README)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=1 pretrain.py \
  --config-name cfg_sft \
  arch/size@arch=XL \
  data.path=/path/to/sft_tokenized \
  resume_from=/path/to/pretrain_checkpoint \
  +checkpoint_path=/path/to/sft_output \
  weights_only_resume_from_ema=true
```

`weights_only_resume_from_ema=true` loads the EMA (exponential moving average) weights from pretraining with a fresh optimizer — this is the clean SFT starting point. The EMA weights are typically slightly smoother than the last training checkpoint.

### The Two Training Paths and Why We Choose Path A

**Path A: HuggingFace model + custom training loop + PEFT LoRA**

The HuggingFace version of HRM-Text-1B has the `hrm_text` architecture class merged into Transformers (requires ≥5.9.0). This means:
- Standard `AutoModelForCausalLM.from_pretrained()` works
- Standard PEFT `LoraConfig` + `get_peft_model()` works
- `token_type_ids` is the PrefixLM control signal at this level

Why LoRA works correctly here: PEFT LoRA modifies weight matrices (`W → W + BA`). It does NOT touch the attention mask computation at all. PrefixLM correctness is entirely handled by the model's own `flash_attention_prefixlm_v2.py` — LoRA is invisible to that mechanism. Confirmed 3-0.

**Path B: Fork pretrain.py with cfg_sft (native packing, no LoRA built-in)**

This uses the repo's native token-count packing (`multipack_sampler.py`) and the `prefix_lens/causal_lens/cu_seqlens` attention system. Pretraining-style correctness guaranteed. But: no built-in LoRA support, requires pre-tokenized data via `prepare_sft_data.py`, and uses FSDP2 (unnecessary overhead for single GPU).

**We use Path A** because: LoRA is essential for memory efficiency (1B model + 5-step BPTT activations = significant memory), and Path A is simpler to implement correctly on a single GPU without FSDP2.

---

## Part 3: Data — Sources, Format, Quality, and the `\boxed{}` Pipeline

### Why These Specific Datasets

**OpenR1-Math-220K (primary, 20K filtered):**
- Generated by DeepSeek-R1 (strong math reasoner) from NuminaMath 1.5 problems
- Every solution prompted with `"Please reason step by step, and put your final answer within \boxed{}."`
- Has a `correctness_math_verify` field — examples already filtered for answer correctness
- Full CoT reasoning traces (up to 16K tokens per generation) followed by a final `\boxed{}` answer
- Why filter to 20K: quality over quantity (confirmed 3-0 — 1-2 sources beat broad mixing by 5%)

**MATH train split (secondary, 7.5K, all examples used):**
- Competition math problems with human-written solutions
- Ground truth for the exact benchmark we're targeting — the SFT data IS the benchmark training set
- Already uses `\boxed{}` throughout

**Why NOT DAPO-Math-17K for SFT:**
This was the dataset that caused the prior experiment failure. It uses AIME-style `"Answer: $N"` integer format with no CoT trace. It was designed as a prompt dataset for RL training (just the problems), not as SFT training data (which needs full solution traces). It remains useful as a source of DAPO prompts (the problems are high-quality) — just never use its answer format.

**Why NOT NuminaMath-CoT as primary:**
860K examples — far too large, quality varies significantly. More importantly, NuminaMath-CoT uses two different answer delimiters: `\boxed{}` AND `■` (the Halmos tombstone symbol, Unicode U+25A0). Math-Verify's LaTeX extraction does not parse `■`. Using NuminaMath directly would silently give reward=0 to every `■`-delimited example. If you need more data, filter NuminaMath to `\boxed{}`-only examples with `re.search(r'\\boxed\{', response)` first.

**Why NOT general instruction data (e.g., ShareGPT, Alpaca):**
NVIDIA/CMU research found that naive mixing of general instruction data with math SFT causes a measurable **-5% average harm on math benchmarks** (not neutral — actually hurts). LoRA already provides protection against forgetting general capabilities, and HRM's diverse pretraining already covers general language understanding. Add more data, make math worse.

### Data Format — Why This Exact Structure

```json
{"condition": "cot,synth", "instruction": "[math problem]", "response": "[step-by-step solution with \\boxed{answer}]"}
```

Every field matters:
- `condition`: special tokens that attend bidirectionally as part of the instruction prefix. Must match the format seen during pretraining. `cot` = produce reasoning steps. `synth` = this is a generated response.
- `instruction`: the math problem. Gets bidirectional (prefix) attention — model reads the full problem before writing.
- `response`: the solution. Gets causal attention and NLL supervision — this is what the model learns to produce.

The `condition` field cannot be omitted or changed arbitrarily. HRM learned specific behaviors conditioned on these special tokens. Using `direct` would tell the model to write a short direct answer; `noisy` would degrade output quality.

### The Complete Preprocessing Pipeline

```python
from datasets import load_dataset
from math_verify import verify, parse
import re, json

def extract_last_boxed(text: str) -> str | None:
    """
    Extract the LAST \\boxed{} occurrence from text.

    WHY LAST (not first or all):
    DeepSeek-R1, which generated OpenR1-Math-220K solutions, frequently produces
    intermediate \\boxed{} values when self-correcting mid-reasoning:
      "...so \\boxed{5}. Wait, that's wrong. Recalculating... \\boxed{7}."
    The LAST \\boxed{} is the final answer. A Math-Verify bug concatenated ALL
    occurrences into a set, causing ~10% benchmark score degradation.
    """
    matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    return matches[-1].group(0) if matches else None

def quality_filter(example):
    # Only keep verified-correct solutions (OpenR1 provides this field)
    if not example.get("correctness_math_verify", False):
        return False
    # Must have a final \boxed{} answer after all self-corrections
    return extract_last_boxed(example.get("solution", "")) is not None

# Load and filter OpenR1
ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
openr1_filtered = ds.filter(quality_filter)
# Take 20K, preferably weighted toward harder problems if difficulty metadata available
openr1_20k = openr1_filtered.select(range(min(20000, len(openr1_filtered))))

# Load MATH train (already clean, already has \boxed{})
math_train = load_dataset("lighteval/MATH", split="train")

def to_hrm(problem: str, solution: str) -> dict:
    return {"condition": "cot,synth", "instruction": problem, "response": solution}

all_examples = (
    [to_hrm(ex["problem"], ex["solution"]) for ex in openr1_20k] +
    [to_hrm(ex["problem"], ex["solution"]) for ex in math_train]
)

# MANDATORY assertion — do not skip under any circumstances
# This is what catches the exact failure mode from the prior experiment
missing_box = [i for i, ex in enumerate(all_examples)
               if not extract_last_boxed(ex["response"])]
assert len(missing_box) == 0, (
    f"CRITICAL: {len(missing_box)} examples missing \\boxed{{}}. "
    f"First 5 indices: {missing_box[:5]}. Fix before training."
)
print(f"✓ {len(all_examples)} examples, 100% \\boxed{{}} verified. Safe to train.")

with open("data/sft_27k.jsonl", "w") as f:
    for ex in all_examples:
        f.write(json.dumps(ex) + "\n")
```

### DAPO Prompt Dataset (~35K problems)

DAPO needs only the problem text — responses are generated online during training. The model generates 8 responses per problem, Math-Verify checks them, and the reward signal flows back.

```python
dapo_prompts = []

# MATH train — Level 3-5 only (where HRM sometimes succeeds, sometimes fails)
# Level 1-2: model already solves these reliably → all-correct batches → zero gradient
# Level 4-5: model rarely solves these → all-wrong batches → zero gradient
# Level 3-4: model sometimes right, sometimes wrong → useful training signal
for ex in math_train:
    level_str = ex.get("level", "Level 0")
    level = int(level_str.split()[-1]) if "Level" in level_str else 0
    if level >= 3:
        dapo_prompts.append({"condition": "cot,synth", "instruction": ex["problem"]})

# OpenR1 hard problems
for ex in openr1_20k:
    dapo_prompts.append({"condition": "cot,synth", "instruction": ex["instruction"]})

# DAPO-Math-17K problem text (problems are excellent; just ignore answer format)
dapo_17k = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
for ex in dapo_17k:
    dapo_prompts.append({"condition": "cot,synth", "instruction": ex["prompt"]})

# Save prompt-only dataset
with open("data/dapo_prompts.jsonl", "w") as f:
    for ex in dapo_prompts:
        f.write(json.dumps(ex) + "\n")
print(f"DAPO prompt pool: {len(dapo_prompts)} problems")
```

---

## Part 4: Stage 1 — SFT with LoRA

### Why SFT Before RL

HRM-Text-1B is a raw base model — no instruction tuning, no RLHF, no alignment. The model exhibits "instruct-like behavior" as a side effect of pretraining on structured Q&A data, but this behavior is inconsistent. Most importantly:

1. The model doesn't reliably produce `\boxed{}` at the end of responses — without this, Math-Verify can't evaluate DAPO rollouts and every reward is 0.
2. The model doesn't reliably format step-by-step solutions in a consistent style — DAPO needs predictable output structure.

SFT teaches format, not knowledge. HRM already knows 56.2% of MATH from pretraining. SFT teaches "always write like this: step-by-step reasoning ending with `\boxed{answer}`."

**Why exactly 1 epoch:** A large-scale empirical study (arXiv 2510.01624, "Quagmires in SFT-RL Post-Training", NeurIPS 2025 Workshop) across 100+ models and 1M+ GPU-hours found that over-SFT actively degrades downstream RL outcomes — even when the post-SFT benchmark scores look good. The mechanism: multiple SFT epochs narrow the model's output distribution (mode collapse), which removes the diversity DAPO needs to explore different reasoning strategies. One epoch is enough to lock in the format without collapsing the distribution.

### LoRA Configuration and Reasoning

**Why LoRA instead of full fine-tuning:**
Full fine-tuning of 1B parameters with 5-step BPTT requires storing activations for all 8 recurrent passes across all layers — roughly 4× the inference memory. LoRA keeps the base model frozen and adds tiny trainable adapter matrices (`delta_W = B·A` where rank=16) to specific weight matrices. This reduces the number of trainable parameters from 1B to ~5M — about 0.5% — while keeping the base model's pretraining intact.

**Why rank 16:**
LoRA's rank controls how many "directions of change" the adapter can represent. Rank 16 = 16 independent ways the model's linear projections can be adjusted. For a model that already knows 56.2% of MATH and just needs to learn output formatting, this is sufficient. The mathematical justification: Aghajanyan et al. (2020) showed that fine-tuning updates for well-pretrained models have low intrinsic dimensionality — they lie in a much lower-dimensional subspace than the full parameter space. Rank 16 captures this adequately for a formatting task.

**Why target attention + MLP projections:**
- **Attention Q/K/V/O:** Controls what information the model retrieves during generation. Adapting these lets the model learn to focus on relevant parts of the problem.
- **MLP gate/up/down:** NeurIPS 2025 ablations (arXiv 2511.06739) show MLP adapters are load-bearing for reasoning tasks — removing them causes worse-than-base performance. Including them is not optional.
- Both H-module and L-module copies: HRM's weight-tying means one adapter is applied N times per forward pass. PEFT handles gradient accumulation across all N calls correctly.

**Why NOT the recurrent state-injection weights (z_L + z_H coupling):**
Two independent reasons:
1. **Theoretical:** In SSM-style recurrent architectures, adapting the linear projection matrices surrounding recurrent modules already captures the same expressiveness as adapting the recurrent state parameters directly (proven in ICML 2025, arXiv 2410.09016). The only thing projection LoRA can't reach is the state transition matrix A — which controls decay rates, not task-specific adaptation.
2. **Practical:** In Mamba-style implementations, recurrent state matrices are bundled in contiguous memory blocks used by fused CUDA kernels. Attaching standard LoRA adapters to them is non-trivial and potentially unstable. For HRM's weight-tied architecture, a single adapter applied N times per forward pass through the recurrent loop compounds in hard-to-predict ways.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

# transformers >= 5.9.0 required
model = AutoModelForCausalLM.from_pretrained(
    "sapientinc/HRM-Text-1B", torch_dtype=torch.bfloat16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")

# Step 1: Inspect actual parameter names (do this before configuring LoRA)
print("Attention/MLP modules found:")
for name, _ in model.named_parameters():
    if any(k in name for k in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']):
        print(f"  {name}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,       # effective LR scaling = alpha/rank = 1.0 (standard)
    lora_dropout=0.05,   # small regularization
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~5M trainable / 1B total (~0.5%)
```

### The Critical Training Loop Detail

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],  # LoRA params only
    lr=2e-5,
    weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))

for batch in dataloader:  # 1 epoch only
    input_ids = batch["input_ids"]
    instruction_lengths = batch["instruction_lengths"]  # length of each instruction

    # ── PrefixLM attention ─────────────────────────────────────────────────────
    # token_type_ids=1 → bidirectional attention (instruction tokens)
    # token_type_ids=0 → causal attention (response tokens)
    # THIS IS WHAT TRL SFTTRAINER CANNOT DO — it applies causal everywhere
    token_type_ids = torch.zeros_like(input_ids)
    for i, L in enumerate(instruction_lengths):
        token_type_ids[i, :L] = 1
    # ──────────────────────────────────────────────────────────────────────────

    # Task-completion objective: train only on response tokens (instruction tokens masked)
    labels = input_ids.clone()
    for i, L in enumerate(instruction_lengths):
        labels[i, :L] = -100  # -100 = ignored by CrossEntropyLoss

    loss = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,  # CRITICAL — must always be passed
        labels=labels,
    ).loss

    loss.backward()
    # max_grad_norm=1.0: aggressive gradient clipping because recurrent BPTT
    # through 5 steps amplifies gradients more than standard transformers
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

## Part 5: Stage 2 — DAPO (Main RL Experiment)

### What Problem DAPO Solves

**The fundamental RL challenge for math:** The model generates hundreds of tokens (a complete solution), then receives a single binary signal: correct or wrong. Which of those hundreds of token decisions caused the error? This is the credit assignment problem. None of the GRPO-family methods solve it perfectly — they use approximations.

**The dominant failure mode at 1B scale:** Entropy collapse. As RL training progresses with standard GRPO, the model's output distribution narrows. Imagine the model as rolling a loaded die to choose each word. Initially many possible rolls. After training, the die becomes more and more biased until the model almost always generates the exact same response to any given prompt. When this happens, GRPO's group-relative advantage (which depends on variance across 8 responses to the same prompt) collapses to zero. Training stalls. The model stopped improving.

**What DAPO fixes:** Two surgical changes to GRPO that delay entropy collapse without changing the fundamental algorithm.

### DAPO Mechanism 1: Clip-Higher

Standard GRPO clips probability changes symmetrically: any token's probability can increase or decrease by at most 20% in one training step (`ε = 0.2`). This is called the PPO clip ratio — it prevents catastrophically large policy updates.

The problem: as the model's distribution narrows, "exploration tokens" — tokens that lead to novel reasoning approaches the model hasn't tried before — have very low current probability. A 20% increase cap means they can only grow from, say, 1% to 1.2%. Not enough to overcome the entropy collapse force.

DAPO uses asymmetric clipping:
```
ε_low  = 0.20   # max decrease per step (same as GRPO)
ε_high = 0.28   # max increase per step (RAISED)
```

Raising the upper bound from 0.20 to 0.28 gives low-probability exploration tokens more room to grow. The mathematical effect: `clip_high` is the dominant driver of entropy increase (confirmed by arXiv 2509.26114 — "Clip-Low Increases Entropy, Clip-High Decreases Entropy" — counterintuitively, the clip values have opposite effects from what you'd guess). The 0.28 setting was confirmed correct in 3-0 adversarial verification from the DAPO paper and veRL official configs.

**Important:** Clip-higher *delays* but does not *prevent* entropy collapse. You still need to monitor entropy and intervene if it collapses.

### DAPO Mechanism 2: Dynamic Sampling

When all 8 responses to a math problem are correct (or all wrong), GRPO computes advantage as `(reward - mean) / std`. If all rewards are identical, `std = 0`, advantage = 0 for everyone, gradient = 0. The model learns nothing from this batch.

Dynamic sampling discards these zero-variance batches and resamples until the batch has mixed correct/wrong responses. It only trains on problems where the model sometimes succeeds and sometimes fails — exactly where learning happens.

**Uncertainty:** Evidence on whether this helps at 1B scale is mixed (2-1 confirmation vs 1-2 contradiction in different papers). Implement it but watch training speed — if it causes the sampler to spin endlessly without finding useful batches (because the model is either very good or very bad at everything), disable it.

### Why β=0.0 (No KL Penalty)

Standard GRPO uses a KL penalty (`β=0.04`) that penalizes the model for drifting too far from the pretrained distribution. The intuition: don't let the model forget how to be a language model while training to be good at math.

DAPO removes this entirely. The reasoning from the DAPO paper: during long-CoT reasoning RL training, the model's distribution necessarily diverges significantly from the initial pretrained distribution as it develops new reasoning strategies. The KL penalty actively fights against this beneficial divergence, limiting how much the model can improve.

This was contested in earlier research (VAPO and ProRL papers argued KL helps for very long training runs). The final confirmed evidence (3-0): DAPO paper + AceReason-Nemotron 1.1 (SOTA 7B math as of June 2025) + Magistral — all remove KL entirely for math RL training. For a one-week run (not 16,000 GPU-hours), β=0.0 is correct.

### DAPO Configuration

```python
epsilon_low      = 0.20    # lower clip bound
epsilon_high     = 0.28    # upper clip bound — the key change from vanilla GRPO
beta_kl          = 0.0     # no KL penalty
group_size       = 8       # responses per prompt (must balance diversity vs compute)
dynamic_sampling = True    # discard zero-variance batches (ablate if training stalls)
reset_interval   = 250     # reset reference model to current policy every 250 steps
                           # Why 250: ProRL shows this prevents KL drift from
                           # accumulating without disrupting learning progress
max_new_tokens   = 4096    # hard limit from HRM context_size=4096
                           # Hard MATH L4-5 problems may truncate — this is expected
```

### Rollout Generation (HF generate, NOT vLLM)

```python
def generate_rollouts(model, prompts, tokenizer, group_size=8, max_new_tokens=4096):
    """
    DAPO rollout generation with correct PrefixLM attention.

    WHY NOT vLLM:
    - PagedAttention assumes causal attention everywhere
    - HRM needs bidirectional attention over instruction tokens
    - vLLM is officially listed as "in progress" on HRM model card
    - Using vLLM would silently apply wrong attention → corrupted rollouts
      → DAPO trains on wrong distribution → model drifts incorrectly

    WHY token_type_ids=ones:
    - All prompt tokens are instruction (prefix) → set all to 1
    - The model automatically switches to causal attention for newly generated tokens
    - Omitting token_type_ids = pure causal = "noticeably worse logits" (HRM model card)
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    # Mark entire prompt as bidirectional prefix (token_type_ids=1)
    token_type_ids = torch.ones_like(inputs.input_ids)

    with torch.no_grad():
        return model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            token_type_ids=token_type_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id,
        )
```

### Reward Function

```python
from math_verify import verify, parse
import re

def extract_last_boxed(text: str) -> str | None:
    """
    Extract the LAST \\boxed{} in the text.
    See Problem 1 Failure C above for why "last" matters.
    """
    matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    return matches[-1].group(0) if matches else None

def reward_fn(response: str, ground_truth: str) -> float:
    """
    Binary correct/wrong reward with small format bonus.

    Returns 0.0 for:
    - No \\boxed{} found (wrong format OR truncated at token cap)
    - \\boxed{} present but answer is wrong

    Returns 1.1 for correct answer with \\boxed{}
    Returns 0.1 for \\boxed{} present but answer wrong
    (The 0.1 format bonus is small enough it can't be gamed without being correct)

    WHY sympy (Math-Verify) not string match:
    At MATH levels 4-5, "1/2", "0.5", and "\\frac{1}{2}" are all the same answer.
    String matching would incorrectly penalize equivalent representations.
    Math-Verify handles symbolic equivalence via sympy.
    """
    last_box = extract_last_boxed(response)
    if last_box is None:
        return 0.0

    try:
        correct = 1.0 if verify(parse(last_box), parse(ground_truth)) else 0.0
    except Exception:
        correct = 0.0

    return correct + 0.1  # max=1.1
```

### What to Monitor and Why

```python
# Log every 50 steps. These are the training health signals:
metrics = {
    "policy_entropy":   ...,  # ENTROPY IS THE MOST IMPORTANT METRIC.
                              # If it monotonically decreases → entropy collapse approaching.
                              # Healthy training: fluctuates but stays relatively stable.
                              # Collapsed: all rollouts become nearly identical → gradient → 0.

    "reward_mean":      ...,  # Should trend upward over training. If flat for 200+ steps:
                              # either entropy has collapsed, or prompts too easy/hard.

    "reward_std":       ...,  # Variance within each batch. If → 0: all rollouts getting
                              # same reward → zero gradient signal → training stalled.

    "boxed_rate":       ...,  # Fraction of rollouts that contain \\boxed{}.
                              # Should stay > 0.8. If < 0.5: check condition flag (must
                              # be cot,synth) and token cap (maybe prompts are too long).

    "truncation_rate":  ...,  # Fraction hitting max_new_tokens without producing \\boxed{}.
                              # If > 0.3: your DAPO prompts include too many hard problems
                              # the model can't solve in 4096 tokens. Filter to Level 3-4.

    "kl_from_ref":      ...,  # KL divergence from reference model. With β=0.0 there's no
                              # explicit penalty, but monitoring reveals if policy is drifting
                              # dangerously far. If > 0.5: consider adding β=0.01 temporarily.
}

# Intervention playbook:
# entropy < (start_entropy - 0.5): raise ε_high from 0.28 → 0.35
# entropy → 0:                     add entropy bonus: loss -= 0.01 × entropy
# boxed_rate < 0.5:                check condition flag; check token cap; check SFT quality
# truncation_rate > 0.3:           filter DAPO prompts to Level 3-4 only
# reward_mean stuck at 0.1:        model has \\boxed{} but all wrong; review DAPO prompt difficulty
```

---

## Part 6: RL Methods — Full Comparison With Reasoning

### ✅ DAPO — Main Experiment

**What it does:** GRPO with two changes — asymmetric clipping (ε_high=0.28 delays entropy collapse) and removal of KL penalty (β=0.0 lets the policy diverge enough to develop new reasoning strategies).

**Why this over vanilla GRPO:** DAPO is a strictly free upgrade — two numbers changed, same infrastructure, same data. There is no reason to use vanilla GRPO over DAPO.

**Evidence:** ε_high=0.28 confirmed 3-0 from DAPO paper arXiv 2503.14476 and veRL official configs. β=0.0 confirmed 3-0 from DAPO paper + AceReason-Nemotron 1.1 + Magistral.

### ✅ REINFORCE++ — Day 7 Comparison

**What it does:** Replaces GRPO's per-prompt advantage normalization (mean and std computed within each group of 8 responses) with global batch normalization (mean and std computed across the entire training batch).

**Why this matters:** GRPO's local normalization has an instability problem: when all 8 responses to a prompt receive similar rewards, the local std approaches zero, causing advantages to blow up (divide by near-zero). REINFORCE++ avoids this by normalizing across the full batch.

**Why it's a comparison, not the main experiment:** DAPO's asymmetric clipping is already confirmed 3-0 and is a simpler implementation change (just config values). REINFORCE++ requires modifying the normalization code. Both are worth testing — the comparison tells you which mechanism (entropy delay vs normalization stability) matters more for HRM specifically.

### ❌ RAFT — Dropped

**What RAFT is:** Rejection Sampling Fine-Tuning. Iterative SFT on the model's own correct rollouts — generate 8 responses per problem, keep the correct ones, fine-tune on them. Repeat.

**Why dropped:** DART-Math (NeurIPS 2024) showed that vanilla rejection sampling generates zero correct responses for 51.1% of Level 5 MATH problems. At 56.2% MATH base accuracy, HRM already reliably solves Level 1-3. RAFT would primarily reinforce problems the model already masters. The performance plateau from prior research: RAFT reaches ~56.1% and then stalls (positive-only training → entropy collapse → the model stops exploring). DAPO reaches ~56.3% and continues improving. Given the starting point at 56.2%, the benefit from RAFT would be negligible and it would consume 1-2 days.

### ❌ PPO — Rejected for Week 1 (Theoretically Interesting)

**What PPO is:** Proximal Policy Optimization. Uses 4 simultaneous model copies: actor (being trained), critic (estimates per-token value), reference model (frozen SFT checkpoint for KL), and reward model (trained separately on preference data).

**The theoretical argument FOR PPO on HRM:** PPO's critic can learn to estimate "how good is this response trajectory at this point?" at every token. For HRM, this could potentially be extended to estimate value at each recurrent state (H-cycle and L-step level), providing more granular credit assignment than DAPO's sequence-level reward. This has never been done for a hierarchical recurrent architecture.

**Why rejected for Week 1:**
1. Requires training a separate reward model (~10-30K preference pairs to construct and train)
2. Requires engineering a critic that understands HRM's recurrent state space — novel research, not implementation
3. 4 model copies simultaneously on a single GPU is very tight even at 1B
4. Empirically, PPO and GRPO achieve comparable performance on math benchmarks at 7B scale — no evidence PPO is worth the overhead

### ❌ DPO — Wrong Tool for Math

**What DPO is:** Direct Preference Optimization. Trains on pairs of responses — "A is preferred over B" — without online generation. Simple, stable, no RL complexity.

**Why wrong for math:** DPO needs *preference* pairs. For math, you have *verifiable truth*: either the answer is correct or it isn't. DPO can't be told "make the model produce more correct answers" — it can only be told "prefer response A over response B." If both A and B are wrong (just one is less wrong), DPO still tries to increase the probability of A. DAPO's binary reward signal (correct=1, wrong=0) is more direct and more appropriate.

**When DPO would be right:** ARC-C, MMLU — multiple choice benchmarks where correct/wrong options create natural preference pairs. We dropped those benchmarks from scope.

### ❌ SDPO — Structurally Incompatible

**What SDPO is:** Self-Distillation Policy Optimization. The same model acts as both student and self-teacher. The student generates a response; the self-teacher is prompted with the same question PLUS rich textual feedback (error messages, execution traces) and generates a corrected response. The KL divergence between their per-token distributions provides dense token-level credit assignment.

**Why incompatible with HRM + math:**
1. SDPO requires *rich textual feedback* as conditioning context — error messages, execution traces, judge evaluations. Math-Verify returns a binary scalar (correct/wrong). There is no rich textual feedback to condition on.
2. SDPO is not validated on the MATH benchmark at all — only chemistry reasoning and coding (which do have rich textual feedback from simulators/compilers).
3. Standalone SDPO underperforms vanilla GRPO at the 10-hour mark (71.1% vs 74.0% — confirmed 3-0 from SRPO paper arXiv 2604.02288). It saturates early then collapses.
4. Not in TRL or veRL — would require custom implementation.

### ❌ GTPO / GRPO-S — All Claims Invalidated

These were proposed as "entropy-weighted token reward" variants that would better handle HRM's non-uniform hierarchical computation. All specific claims about GTPO (+29.4pp AIME gain, entropy rebound mechanism, computational efficiency of GRPO-S) were independently refuted 0-3 in adversarial verification across three separate research passes. Do not pursue.

### ❌ General Data Mixing in SFT — Actively Harmful

The idea: mix some general instruction data (ShareGPT, Alpaca, etc.) into SFT to prevent forgetting general capabilities.

Why rejected: NVIDIA/CMU research found that naive scaling of mixed-quality SFT data causes a measurable **-5% harm on math benchmarks** — not neutral, actually worse. LoRA already provides meaningful protection against forgetting (Biderman et al. TMLR 2024: LoRA forgets less than full fine-tuning). HRM's pretraining on 40B tokens of diverse data already handles general language capability. Adding general data to math SFT dilutes the quality signal and actively hurts the math target.

---

## Part 7: Hardware & Infrastructure

### GPU Requirements and Why

**Why H100/H200 specifically (not A100, not L40S):**

HRM's training codebase uses FlashAttention 3 (`flash_attention_prefixlm_v2.py`). FA3 is implemented using Hopper-specific tensor core operations (warp-specialized pipeline, FP8 support) that physically do not exist on:
- A100 (Ampere architecture): uses FA2 only. FA2 is functionally equivalent but slower (~1.5-2× slower than FA3).
- L40S (Ada Lovelace): uses FA2 only. Additionally, L40S has no NVLink — inter-GPU communication over PCIe is ~10× slower. Not suitable for distributed training.

**FA3 availability check — Day 1 action:**
```bash
grep -n "flash_attn_func\|flash_attn_varlen\|FA2\|FA3\|hopper" models/flash_attention_prefixlm_v2.py
```
If there's an FA2 fallback path, A100 becomes viable (slower but functional). If not, H100/H200 is required.

**Memory footprint (single H200, 141GB):**

| Component | Memory |
|-----------|--------|
| Model weights (1B, bf16) | ~2 GB |
| LoRA adapters (rank 16, ~5M params) | ~64 MB |
| Adam optimizer states (LoRA only: m + v per param) | ~128 MB |
| BPTT activations (5 recurrent steps, 4096 ctx) | ~15–25 GB |
| KV cache during DAPO rollout generation | ~5–10 GB |
| **Total** | **~22–37 GB** |

H200 141GB has roughly 4× headroom. An H100 80GB would also fit comfortably. No FSDP2 needed — that's for distributing a model that doesn't fit on one GPU.

### Hyak Cluster Configuration

**The gpu-h200 partition:** User-confirmed H200 nodes accessible via `-p gpu-h200` on Hyak Klone. This is a **ckpt (checkpoint) partition** — preemptible. When users with priority access (condo owners of those nodes) submit jobs, yours gets preempted.

**How preemption works on ckpt:**
- SLURM sends SIGTERM → trainer saves checkpoint (handled by MAX_RUNTIME budget in the script)
- Job is requeued in the ckpt queue
- When resources become available, job restarts from latest checkpoint
- The self-resubmit loop in the sbatch script handles this automatically

**ckpt GPU walltime limit: 8-9 hours** (increased from 4-5 hours in April 2024 maintenance). This doesn't mean jobs run for only 8 hours — the self-resubmit loop chains jobs end-to-end. SFT (1-2 days of compute) across 4-6 consecutive 8-hour slots.

**Storage:** `/mmfs1/gscratch/intelligentsystems/$USER/` — use this for checkpoints, training data, and HF cache. It's on Spectrum Scale (GPFS), 2.4PB total, 100Gbps bandwidth. Fast enough for checkpoint save/load.

### FlashAttention 3 Installation

FA3 is NOT available via `pip install flash-attn` — that command installs FA2. FA3 requires compiling the Hopper-specific kernel from source:

```bash
# Prerequisites: CUDA >= 12.3, Hopper GPU available during build, ninja
pip install ninja  # dramatically speeds up compilation

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=4 python setup.py install  # ~30 min without ninja, ~5 min with ninja

# Verify the kernel loads correctly in HRM's context:
cd /path/to/HRM-Text
python -c "from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm; print('FA3 PrefixLM kernel loaded OK')"
```

**Alternative:** FlashAttention 4 (`pip install --pre flash-attn-4`) — JIT-compiled, no CUDA compilation needed, requires Hopper + CUDA >= 12.3. Newer and potentially faster, but less battle-tested with HRM's specific kernel.

### Sbatch Script

```bash
#!/bin/bash
#SBATCH --job-name=hrm-training
#SBATCH --partition=gpu-h200            # H200 ckpt partition on Klone (user-confirmed)
#SBATCH --account=intelligentsystems-ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:h200:1              # Single H200 — no FSDP2 needed at 1B scale
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00                 # ckpt GPU walltime: 8-9h max
                                       # Self-resubmit loop chains jobs for longer runs
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue                      # Auto-requeue when preempted
#SBATCH --open-mode=append             # Append logs across resubmits
#SBATCH --signal=B:USR1@180            # 3-min signal before walltime kill → triggers checkpoint
# ─────────────────────────────────────────────────────────────────
# PREEMPTION HANDLING:
#   - --requeue: SLURM requeues the job when preempted by priority users
#   - --signal: gives trainer 180s warning to save a clean checkpoint
#   - MAX_RUNTIME: trainer exits 15min before walltime, saves checkpoint
#   - Self-resubmit loop: if clean exit + checkpoint exists → sbatch again
#   - Result: training continues seamlessly across preemptions
#
# SINGLE GPU:
#   - No FSDP2 (designed for multi-GPU pretraining, adds overhead at 1B)
#   - torchrun with nproc_per_node=1 for consistent launch interface
#
# NOT USING:
#   - TRL SFTTrainer: causal-only, corrupts PrefixLM bidirectional attention
#   - vLLM: PagedAttention causal-only, listed "in progress" on HRM model card
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_PATH=${ENV_PATH:-/mmfs1/gscratch/intelligentsystems/$USER/hrm_env}
REPO_ROOT=$(pwd)
JOBID=${SLURM_JOB_ID:-local-$(date +%s)}
LOG_DIR="${REPO_ROOT}/logs/run-${JOBID}"
mkdir -p "${LOG_DIR}"

OUT_DIR=${OUT_DIR:-/mmfs1/gscratch/intelligentsystems/$USER/hrm_runs/sft}
mkdir -p "${OUT_DIR}"

export ENV_PATH OUT_DIR
echo "[sbatch] job=${JOBID} node=$(hostname) partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi -L || true

export PATH="${ENV_PATH}/bin:${PATH}"
export HF_HOME=${HF_HOME:-/mmfs1/gscratch/intelligentsystems/$USER/hf_cache}
mkdir -p "${HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# expandable_segments: prevents CUDA allocator fragmentation during long RL runs
# (after many rollout+update cycles, allocator fragments badly without this)

# Wall-clock budget: derive seconds until SLURM hard-kills the job
# Exit 15min early so trainer can save a clean checkpoint before SIGKILL
MARGIN=${MARGIN:-900}
MAX_RUNTIME=0
END_RAW=$(scontrol show job "${JOBID}" -o 2>/dev/null \
          | grep -oE 'EndTime=[^ ]+' | cut -d= -f2 || true)
if [[ -n "${END_RAW}" && "${END_RAW}" != "Unknown" ]]; then
  END_EPOCH=$(date -d "${END_RAW}" +%s 2>/dev/null || echo 0)
  NOW_EPOCH=$(date +%s)
  if [[ "${END_EPOCH}" -gt "${NOW_EPOCH}" ]]; then
    MAX_RUNTIME=$(( END_EPOCH - NOW_EPOCH - MARGIN ))
    [[ "${MAX_RUNTIME}" -lt 60 ]] && MAX_RUNTIME=60
  fi
fi
echo "[sbatch] wall-clock budget: MAX_RUNTIME=${MAX_RUNTIME}s"

# Resume from latest checkpoint if one exists
LATEST=$(ls -td "${OUT_DIR}"/checkpoint-* 2>/dev/null | head -n1 || true)
RESUME_ARGS=()
[[ -n "${LATEST}" ]] && {
  echo "[sbatch] resuming from ${LATEST}"
  RESUME_ARGS=(--resume_from_checkpoint "${LATEST}")
}

STAGE=${STAGE:-sft}
case "${STAGE}" in
  sft)
    TRAIN_SCRIPT="sft/train_hrm_sft.py"
    # Custom SFT script: HF model + PEFT LoRA + custom loop with token_type_ids
    EXTRA_ARGS="--dataset_path data/sft_27k.jsonl \
                --num_train_epochs 1 \
                --max_seq_length 4096 \
                --learning_rate 2e-5 \
                --lora_r 16 --lora_alpha 16 \
                --lora_dropout 0.05"
    ;;
  dapo)
    TRAIN_SCRIPT="rl/train_dapo.py"
    EXTRA_ARGS="--epsilon_low 0.20 --epsilon_high 0.28 --beta_kl 0.0 \
                --group_size 8 --dynamic_sampling true \
                --ref_reset_steps 250 --max_steps 1000 \
                --max_new_tokens 4096 \
                --condition cot,synth \
                --use_hf_generate true"   # NOT vLLM
    ;;
  reinforce)
    TRAIN_SCRIPT="rl/train_reinforce_pp.py"
    EXTRA_ARGS="--global_batch_normalization true --max_steps 1000 \
                --max_new_tokens 4096 --use_hf_generate true"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}"; exit 1 ;;
esac

torchrun --standalone --nproc_per_node=1 \
  "${TRAIN_SCRIPT}" \
  --output_dir "${OUT_DIR}" \
  --max_runtime_seconds "${MAX_RUNTIME}" \
  --bf16 true \
  --max_grad_norm 1.0 \
  ${EXTRA_ARGS} \
  "${RESUME_ARGS[@]}" \
  "$@" \
  > >(tee -a "${LOG_DIR}/train.log") 2>&1 &

TORCHRUN_PID=$!
# Forward SIGUSR1 (the 3-min warning signal) to the trainer process
trap 'echo "[sbatch] USR1 → forwarding to trainer"; kill -USR1 "${TORCHRUN_PID}" 2>/dev/null || true' USR1

# Wait loop: handle signal interruption without exiting early
RC=""
while [[ -z "${RC}" ]]; do
  if wait "${TORCHRUN_PID}"; then RC=0
  else
    code=$?; kill -0 "${TORCHRUN_PID}" 2>/dev/null && continue; RC=${code}
  fi
done
echo "[sbatch] torchrun exit=${RC}"

# Self-resubmit: clean exit (RC=0) + checkpoint = preemption, requeue
RESUBMIT_MAX=${RL_RESUBMIT_MAX:-20}
RESUBMIT_N=${RL_RESUBMIT_N:-0}
if [[ -f "${OUT_DIR}/COMPLETE" ]]; then
  echo "[sbatch] COMPLETE — training finished, not resubmitting."
elif [[ "${RC}" -eq 0 ]] && ls -d "${OUT_DIR}"/checkpoint-* >/dev/null 2>&1; then
  if [[ "${RESUBMIT_N}" -ge "${RESUBMIT_MAX}" ]]; then
    echo "[sbatch] hit RESUBMIT_MAX=${RESUBMIT_MAX}; stop chain. Resubmit manually to continue."
  else
    echo "[sbatch] incomplete + clean exit → resubmitting (chain ${RESUBMIT_N}→$((RESUBMIT_N+1)))"
    sbatch --export=ALL,RL_RESUBMIT_N=$((RESUBMIT_N+1)),ENV_PATH="${ENV_PATH}",OUT_DIR="${OUT_DIR}",STAGE="${STAGE}" \
           "${REPO_ROOT}/submit_hrm.sbatch" "$@"
  fi
else
  echo "[sbatch] exit=${RC} with no resumable checkpoint — check logs before resubmitting."
fi
exit "${RC}"
```

**Usage:**
```bash
# Confirm partition access first
sinfo -p gpu-h200
hyakalloc   # shows your account's GPU allocations

# Stage 1: SFT (Days 1-2)
OUT_DIR=/mmfs1/gscratch/intelligentsystems/$USER/hrm_runs/sft \
STAGE=sft sbatch submit_hrm.sbatch

# Stage 2: DAPO (Days 3-6) — start from SFT checkpoint
OUT_DIR=/mmfs1/gscratch/intelligentsystems/$USER/hrm_runs/dapo \
STAGE=dapo sbatch submit_hrm.sbatch \
  --init_model /mmfs1/gscratch/.../hrm_runs/sft/checkpoint-final

# Stage 3: REINFORCE++ comparison (Day 7) — same SFT checkpoint, clean comparison
OUT_DIR=/mmfs1/gscratch/intelligentsystems/$USER/hrm_runs/reinforce \
STAGE=reinforce sbatch submit_hrm.sbatch \
  --init_model /mmfs1/gscratch/.../hrm_runs/sft/checkpoint-final
```

---

## Part 7.5: Complete Script Skeletons (For New Implementation Chat)

### `sft/train_hrm_sft.py` — Full SFT Script Skeleton

```python
#!/usr/bin/env python3
"""
HRM-Text-1B SFT with LoRA
Uses HuggingFace model + PEFT LoRA + custom training loop.
CRITICAL: token_type_ids must be passed to model.forward() for correct PrefixLM attention.
"""
import argparse, json, os, signal, sys, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="sapientinc/HRM-Text-1B")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_runtime_seconds", type=int, default=0)
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()

class MathSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        with open(path) as f:
            for line in f:
                ex = json.loads(line.strip())
                # Format: build the full prompt string
                # condition flag + instruction = prefix (bidirectional)
                # response = causal (supervised)
                condition = ex["condition"]   # e.g., "cot,synth"
                instruction = ex["instruction"]
                response = ex["response"]

                # Build full prompt (condition is embedded as special tokens by tokenizer)
                prompt = f"<|im_start|>{condition}{instruction}<|im_end|>"
                full = prompt + response + tokenizer.eos_token

                # Tokenize separately to find instruction boundary
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_ids = tokenizer.encode(full, add_special_tokens=False)

                if len(full_ids) > max_seq_length:
                    full_ids = full_ids[:max_seq_length]

                instruction_len = len(prompt_ids)
                self.examples.append({
                    "input_ids": full_ids,
                    "instruction_length": instruction_len,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, tokenizer, max_seq_length):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    max_len = min(max_len, max_seq_length)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    token_type_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        ids = ex["input_ids"][:max_len]
        inst_len = min(ex["instruction_length"], max_len)
        L = len(ids)

        input_ids[i, :L] = torch.tensor(ids)
        attention_mask[i, :L] = 1
        token_type_ids[i, :inst_len] = 1          # instruction = bidirectional prefix
        labels[i, inst_len:L] = torch.tensor(ids[inst_len:L])  # only response tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).cuda()

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Adjust these after inspecting actual parameter names:
        # [n for n,_ in model.named_parameters() if 'proj' in n or 'gate' in n]
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = MathSFTDataset(args.dataset_path, tokenizer, args.max_seq_length)
    from functools import partial
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_seq_length=args.max_seq_length),
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * args.num_train_epochs
    )

    # Wall-clock budget handling
    start_time = time.time()
    stop_requested = False
    def handle_usr1(sig, frame):
        nonlocal stop_requested
        print("[train] SIGUSR1 received — will checkpoint and exit after current step")
        stop_requested = True
    signal.signal(signal.SIGUSR1, handle_usr1)

    model.train()
    global_step = 0

    for epoch in range(args.num_train_epochs):
        for batch in loader:
            if stop_requested:
                break
            if args.max_runtime_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed > args.max_runtime_seconds:
                    print(f"[train] MAX_RUNTIME reached ({elapsed:.0f}s) — saving checkpoint")
                    stop_requested = True
                    break

            batch = {k: v.cuda() for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],  # CRITICAL — correct PrefixLM
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 100 == 0:
                print(f"[train] step={global_step} loss={loss.item():.4f} "
                      f"elapsed={time.time()-start_time:.0f}s")

        if stop_requested:
            break

    # Save checkpoint
    ckpt_path = Path(args.output_dir) / f"checkpoint-{global_step}"
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    print(f"[train] Saved checkpoint to {ckpt_path}")

    if not stop_requested:
        (Path(args.output_dir) / "COMPLETE").touch()
        print("[train] Training COMPLETE")

if __name__ == "__main__":
    main()
```

### `rl/train_dapo.py` — DAPO Training Script Skeleton

```python
#!/usr/bin/env python3
"""
DAPO training for HRM-Text-1B.
Uses HF generate() with token_type_ids for correct PrefixLM rollout generation.
NOT using vLLM (incompatible with HRM's PrefixLM attention).
"""
import argparse, json, os, re, signal, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import verify, parse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)  # SFT checkpoint
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_path", required=True)         # DAPO prompt JSONL
    p.add_argument("--epsilon_low", type=float, default=0.20)
    p.add_argument("--epsilon_high", type=float, default=0.28)
    p.add_argument("--beta_kl", type=float, default=0.0)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--dynamic_sampling", action="store_true")
    p.add_argument("--ref_reset_steps", type=int, default=250)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--condition", default="cot,synth")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_runtime_seconds", type=int, default=0)
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()

def extract_last_boxed(text: str) -> str | None:
    """Extract LAST \\boxed{} — intermediate values from self-correction ignored."""
    matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    return matches[-1].group(0) if matches else None

def reward_fn(response: str, ground_truth: str) -> float:
    last_box = extract_last_boxed(response)
    if last_box is None:
        return 0.0
    try:
        correct = 1.0 if verify(parse(last_box), parse(ground_truth)) else 0.0
    except Exception:
        correct = 0.0
    return correct + 0.1  # 0.1 format bonus

def generate_rollouts(model, prompt_texts, tokenizer, group_size, max_new_tokens, device):
    """Generate rollouts with correct PrefixLM attention. NOT vLLM."""
    inputs = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, truncation=True,
        max_length=512  # prompt shouldn't be too long to leave room for response
    ).to(device)
    # All prompt tokens are instruction prefix → bidirectional attention
    token_type_ids = torch.ones_like(inputs.input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            token_type_ids=token_type_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs.input_ids.shape[1]
    # Only return the generated response (not the prompt)
    return outputs[:, prompt_len:]

def compute_dapo_loss(model, ref_model, prompt_input_ids, prompt_token_type_ids,
                      responses, rewards, tokenizer, epsilon_low, epsilon_high, beta_kl):
    """
    DAPO loss computation.
    Asymmetric clipping: epsilon_high > epsilon_low.
    No KL penalty (beta_kl=0.0).
    """
    B = len(rewards)  # batch_size * group_size
    G = 8  # group_size (hardcoded for clarity; use epsilon_high arg)

    # Normalize rewards within each group (group-relative advantage)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    advantages = []
    for i in range(0, B, G):
        group = rewards_tensor[i:i+G]
        mean, std = group.mean(), group.std()
        if std < 1e-8:  # zero-variance batch (dynamic sampling should prevent this)
            advantages.extend([torch.tensor(0.0)] * G)
        else:
            advantages.extend(((group - mean) / std).tolist())
    advantages = torch.tensor(advantages).to(model.device)

    # TODO: Implement full DAPO policy gradient loss with asymmetric clipping
    # This skeleton shows the structure; fill in token-level log-prob computation
    # following the DAPO paper (arXiv 2503.14476) equations

    raise NotImplementedError(
        "Implement full DAPO loss here. Key steps:\n"
        "1. Compute log-probs of responses under current policy\n"
        "2. Compute log-probs of responses under old policy (from rollout)\n"
        "3. Compute ratio = exp(new_logprobs - old_logprobs)\n"
        "4. Apply asymmetric clipping: clip(ratio, 1-ε_low, 1+ε_high)\n"
        "5. loss = -min(ratio*adv, clipped_ratio*adv).mean()\n"
        "6. If beta_kl > 0: add KL penalty term\n"
        "Reference: veRL DAPO implementation for exact tensor operations"
    )

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).to(device)

    # Reference model (frozen copy for KL; even at beta_kl=0, used for monitoring)
    import copy
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Load prompts
    prompts = []
    with open(args.dataset_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            condition = ex.get("condition", "cot,synth")
            instruction = ex["instruction"]
            prompt_text = f"<|im_start|>{condition}{instruction}<|im_end|>"
            ground_truth = ex.get("answer", ex.get("response", ""))
            prompts.append({"prompt": prompt_text, "answer": ground_truth})

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    start_time = time.time()
    stop_requested = False
    def handle_usr1(sig, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, handle_usr1)

    import random
    step = 0
    while step < args.max_steps and not stop_requested:
        if args.max_runtime_seconds > 0 and time.time() - start_time > args.max_runtime_seconds:
            print("[dapo] MAX_RUNTIME reached — saving checkpoint")
            break

        # Sample a batch of prompts
        batch_prompts = random.sample(prompts, min(4, len(prompts)))
        prompt_texts = [ex["prompt"] for ex in batch_prompts]
        ground_truths = [ex["answer"] for ex in batch_prompts]

        # Generate rollouts (8 responses per prompt)
        responses_ids = generate_rollouts(
            model, prompt_texts, tokenizer, args.group_size, args.max_new_tokens, device
        )
        responses_text = [
            tokenizer.decode(r, skip_special_tokens=True) for r in responses_ids
        ]

        # Compute rewards
        rewards = []
        for i, response in enumerate(responses_text):
            gt = ground_truths[i // args.group_size]
            rewards.append(reward_fn(response, gt))

        # Monitoring
        reward_arr = torch.tensor(rewards)
        reward_std = reward_arr.std().item()
        boxed_rate = sum(1 for r in responses_text if extract_last_boxed(r)) / len(responses_text)
        truncated = sum(1 for r_id in responses_ids if r_id[-1].item() != tokenizer.eos_token_id)

        if args.dynamic_sampling and reward_std < 1e-6:
            print(f"[dapo] step={step} Zero-variance batch, skipping (dynamic sampling)")
            continue

        # DAPO loss and update
        # (Full implementation needed — see NotImplementedError in compute_dapo_loss)
        # loss = compute_dapo_loss(...)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # optimizer.step()
        # optimizer.zero_grad()

        step += 1
        print(f"[dapo] step={step}/{args.max_steps} "
              f"reward_mean={reward_arr.mean():.3f} reward_std={reward_std:.3f} "
              f"boxed_rate={boxed_rate:.2f} truncated={truncated}/{len(responses_text)}")

        # Reference model reset (ProRL)
        if step % args.ref_reset_steps == 0:
            print(f"[dapo] Resetting reference model at step {step}")
            ref_model = copy.deepcopy(model).eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

    # Save
    ckpt_path = Path(args.output_dir) / f"checkpoint-{step}"
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    print(f"[dapo] Saved to {ckpt_path}")

    if step >= args.max_steps:
        (Path(args.output_dir) / "COMPLETE").touch()

if __name__ == "__main__":
    main()
```

### `data/prepare_data.py` — Data Preprocessing Script

```python
#!/usr/bin/env python3
"""
Prepare SFT and DAPO datasets for HRM-Text-1B post-training.
Handles the \\boxed{} extraction problem (take LAST, not first or all).
"""
import re, json, argparse
from datasets import load_dataset

def extract_last_boxed(text: str) -> str | None:
    matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    return matches[-1].group(0) if matches else None

def quality_filter_openr1(example):
    """Filter OpenR1-Math-220K for quality."""
    if not example.get("correctness_math_verify", False):
        return False
    return extract_last_boxed(example.get("solution", "")) is not None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_sft", default="data/sft_27k.jsonl")
    p.add_argument("--output_dapo", default="data/dapo_prompts.jsonl")
    p.add_argument("--openr1_size", type=int, default=20000)
    args = p.parse_args()

    sft_examples = []
    dapo_prompts = []

    # OpenR1-Math-220K
    print("Loading OpenR1-Math-220K...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    filtered = ds.filter(quality_filter_openr1)
    openr1_sample = filtered.select(range(min(args.openr1_size, len(filtered))))
    for ex in openr1_sample:
        sft_examples.append({
            "condition": "cot,synth",
            "instruction": ex["problem"],
            "response": ex["solution"],
        })
        dapo_prompts.append({
            "condition": "cot,synth",
            "instruction": ex["problem"],
            "answer": ex["answer"],
        })
    print(f"OpenR1: {len(openr1_sample)} examples after filtering")

    # MATH train split
    print("Loading MATH train split...")
    math_train = load_dataset("lighteval/MATH", split="train")
    for ex in math_train:
        sft_examples.append({
            "condition": "cot,synth",
            "instruction": ex["problem"],
            "response": ex["solution"],
        })
        level = int(ex.get("level","Level 0").split()[-1]) if "Level" in ex.get("level","") else 0
        if level >= 3:
            dapo_prompts.append({
                "condition": "cot,synth",
                "instruction": ex["problem"],
                "answer": ex["solution"],  # MATH train has full solution; extract boxed for eval
            })
    print(f"MATH train: {len(math_train)} examples")

    # MANDATORY assertion — do not skip
    missing = [i for i, ex in enumerate(sft_examples) if not extract_last_boxed(ex["response"])]
    assert len(missing) == 0, f"CRITICAL: {len(missing)} SFT examples missing \\boxed{{}}"
    print(f"✓ SFT: {len(sft_examples)} examples, 100% \\boxed{{}} verified")

    # DAPO-Math-17K (problems only for DAPO prompt pool)
    print("Loading DAPO-Math-17K...")
    dapo_17k = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    for ex in dapo_17k:
        dapo_prompts.append({
            "condition": "cot,synth",
            "instruction": ex["prompt"],
            "answer": str(ex.get("answer", "")),
        })

    # Save
    import os
    os.makedirs("data", exist_ok=True)
    with open(args.output_sft, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(sft_examples)} SFT examples → {args.output_sft}")

    with open(args.output_dapo, "w") as f:
        for ex in dapo_prompts:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(dapo_prompts)} DAPO prompts → {args.output_dapo}")

if __name__ == "__main__":
    main()
```

### MATH Benchmark Evaluation Helper

```python
# eval/evaluate_math.py
import re, json
from datasets import load_dataset
from math_verify import verify, parse

def extract_last_boxed(text):
    matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    return matches[-1].group(0) if matches else None

def evaluate_on_math(model, tokenizer, n_samples=500, device="cuda"):
    """Quick MATH benchmark evaluation."""
    import torch, random
    math_test = load_dataset("lighteval/MATH", split="test")
    samples = random.sample(list(math_test), min(n_samples, len(math_test)))

    correct = 0
    no_boxed = 0
    for ex in samples:
        prompt = f"<|im_start|>cot,synth{ex['problem']}<|im_end|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=4096, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        last_box = extract_last_boxed(response)
        if last_box is None:
            no_boxed += 1
            continue
        try:
            if verify(parse(last_box), parse(ex["solution"])):
                correct += 1
        except Exception:
            pass

    total = len(samples)
    print(f"MATH eval: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"  No \\boxed{{}}: {no_boxed}/{total} ({100*no_boxed/total:.1f}%)")
    return correct / total

if __name__ == "__main__":
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", trust_remote_code=True).cuda()
    evaluate_on_math(model, tokenizer, n_samples=500)
```

---

## Part 8: One-Week Timeline

```
Day 1 (Monday) — Setup + Data Prep:
├── Confirm access: sinfo -p gpu-h200 && hyakalloc
├── Check transformers version: python -c "import transformers; print(transformers.__version__)"
│   → Needs >= 5.9.0 for hrm_text architecture class
├── Install FA3: cd flash-attention/hopper && python setup.py install
│   OR: pip install --pre flash-attn-4
├── Verify FA3 loads: python -c "from models.flash_attention_prefixlm_v2 import ..."
├── Run data pipeline:
│   → Download OpenR1-Math-220K
│   → Filter: correctness_math_verify=True
│   → Extract last \\boxed{} from each solution
│   → Assert 100% \\boxed{} coverage — do NOT skip this
│   → Add MATH train split (7.5K)
│   → Save as data/sft_27k.jsonl
├── Inspect HRM parameter names for LoRA target confirmation:
│   model = AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B")
│   [print(n) for n,_ in model.named_parameters() if 'proj' in n or 'gate' in n]
└── Submit SFT: OUT_DIR=.../sft STAGE=sft sbatch submit_hrm.sbatch

Day 2 (Tuesday) — SFT + Prep:
├── Monitor SFT every hour: check logs/run-JOBID/train.log
├── If preempted: check squeue — job auto-requeues
├── SFT completes (~1 epoch on 27.5K examples ≈ 4-6 GPU-hours)
├── Eval post-SFT: run MATH benchmark on 200-500 problems
│   → Record baseline: base=56.2%, post-SFT=expected 58-62%
├── Build DAPO prompt pool: ~35K problems, Level 3-5 weighted
│   → MATH train L3-5 + OpenR1 filtered + DAPO-Math-17K problems
└── Spot-check rollout quality: generate 5 responses, verify \\boxed{} at end

Days 3–5 (Wednesday–Friday) — DAPO:
├── Submit DAPO: OUT_DIR=.../dapo STAGE=dapo sbatch submit_hrm.sbatch
├── Monitor every 50 steps (check W&B or training log):
│   → policy_entropy: must not monotonically decrease
│   → boxed_rate: keep > 0.8
│   → reward_mean: should trend upward
│   → truncation_rate: if > 0.3, filter to Level 3-4 prompts only
├── Reference model reset fires automatically every 250 steps
├── Preemptions handled automatically by self-resubmit loop
│   → Check squeue occasionally; job should always be running or queued
└── Intervention if entropy collapses: raise ε_high to 0.35

Day 6 (Saturday) — Evaluation:
├── Evaluate DAPO checkpoints at step 500 and step 1000 on full MATH
├── Record: base 56.2% → post-SFT ~60% → post-DAPO (target: 62-68%)
└── Prepare REINFORCE++ config (change normalization logic in training script)

Day 7 (Sunday) — Comparison:
├── Submit REINFORCE++ from same SFT checkpoint (clean A/B comparison)
├── Final full MATH eval on DAPO vs REINFORCE++ best checkpoints
└── Document: which method gains more, at what compute cost
```

---

## Part 9: Complete Decision Table

| Decision | Value | Confidence | Full Reasoning |
|----------|-------|------------|----------------|
| SFT framework | Custom loop with `token_type_ids` | **High (3-0)** | TRL SFTTrainer is causal-only — applies wrong attention to instruction tokens. HRM's own pretrain.py with cfg_sft is the alternative. |
| Rollout generation | HF `generate()` + `token_type_ids` | **High (3-0)** | vLLM officially "in progress" on HRM model card — PagedAttention assumes causal attention, cannot apply PrefixLM bidirectional mask. |
| Primary SFT data | OpenR1-Math-220K (filtered 20K) | High | Only dataset with full CoT traces, correctness verification, and reliable `\boxed{}`. DeepSeek-R1 generated. |
| Secondary SFT data | MATH train split (7.5K) | High | Ground truth for the exact benchmark we're targeting. Always has `\boxed{}`. |
| DAPO-Math-17K for SFT | ❌ Dropped | **High** | Uses `"Answer: $N"` format. No `\boxed{}`. No CoT traces. This caused the prior experiment failure. |
| NuminaMath-CoT as primary | ❌ Dropped | **High** | Uses `■` delimiter that Math-Verify cannot parse. Silent reward failures. |
| `\boxed{}` extraction | LAST occurrence only | **High (3-0)** | Intermediate `\boxed{}` from DeepSeek-R1 self-correction. Math-Verify bug concatenated all → ~10% score degradation. |
| 100% `\boxed{}` assertion | Mandatory before training | **High** | Prior failure: SFT dataset had no `\boxed{}` → all DAPO rewards = 0 → nothing trained. |
| SFT epochs | 1 only | High (2-1) | Over-SFT (2+ epochs) narrows model distribution → reduces DAPO exploration → worse RL outcomes (arXiv 2510.01624, NeurIPS 2025). |
| max_seq_length | 4096 | **High** | HRM pretraining context_size=4096. Prior 2048 cap truncated solutions before `\boxed{}`. |
| Condition flag | `cot,synth` throughout | **High (3-0)** | `cot` = step-by-step format. `synth` = synthetic data. DAPO designed for long-CoT (paper uses 20K tokens). Switching to `direct` after `cot` SFT = distribution mismatch. |
| max_new_tokens | 4096 | High | Hard limit from HRM context_size. Level 4-5 hard problems may truncate — expected. reward=0 for truncated responses is correct. |
| PEFT LoRA on HF model | ✅ Architecturally correct | **High (3-0)** | LoRA modifies weight matrices only, not attention mask logic. PrefixLM correctness entirely handled by model's own code via `token_type_ids`. `hrm_text` class merged into Transformers >= 5.9.0. |
| LoRA rank | 16 | Medium | Standard for 1B SFT. Formatting task ≈ low intrinsic dimensionality ≈ low rank sufficient. Ablate to 32 if MATH plateaus after SFT. |
| LoRA targets | Attn+MLP projections, not recurrent weights | Medium (2-1) | MLP adapters load-bearing for reasoning (NeurIPS 2025). Projection LoRA = equivalent expressiveness to recurrent state LoRA (ICML 2025). Recurrent weight compounding and CUDA kernel constraints make state-weight LoRA risky. Plan ablation. |
| No FSDP2 | Single GPU | **High** | FSDP2 designed for sharding models that don't fit on one GPU. 1B model fits on H200 with 4× headroom — FSDP2 adds overhead with no benefit. |
| DAPO ε_high | 0.28 | **High (3-0)** | DAPO paper + veRL official configs. Delays entropy collapse by giving exploration tokens more room to increase probability. |
| DAPO β | 0.0 | **High (3-0)** | DAPO paper + AceReason-Nemotron 1.1 + Magistral — all remove KL for math RL. KL penalty limits policy divergence needed for new reasoning strategies. |
| Reference model resets | Every 250 steps | Medium | ProRL research. Prevents accumulated KL drift while allowing continued improvement. Primary designed for long training runs; still beneficial at 1K steps. |
| Hardware | gpu-h200 (H200, ckpt) | **User-confirmed** | Preemptible. Self-resubmit loop in sbatch script handles preemption automatically. |
| RAFT warm-start | ❌ Dropped | High (2-1) | 51.1% Level-5 miss rate at 56%+ base (DART-Math, NeurIPS 2024). HRM already near RAFT's plateau. No meaningful benefit in 1 week. |
| General data mixing | ❌ No | High (3-0) | NVIDIA/CMU: -5% math harm from naive mixed-quality SFT scaling. LoRA already protects against forgetting. |
| SDPO | ❌ Incompatible | High (3-0) | Needs rich textual feedback (error messages) as self-teacher signal. Math-Verify gives binary scalar only. Also underperforms GRPO at 10-hour mark. |
| GTPO / GRPO-S | ❌ Invalidated | High | All specific claims (performance gains, mechanism, implementation) refuted 0-3 in adversarial verification across 3 separate research passes. |
| PPO | ❌ Week 2+ | High | 4 model copies + training a reward model + engineering a recurrent-state critic = weeks of work. Theoretically interesting for HRM specifically, but not in 1 week. |
| DPO | ❌ Wrong tool | High | Needs preference pairs (A > B). Math has verifiable binary truth (right/wrong). Binary reward in DAPO is strictly more informative and direct. |

---

## Sources

### HRM Architecture & Codebase
- [HRM-Text paper (arXiv 2605.20613)](https://arxiv.org/abs/2605.20613)
- [sapientinc/HRM-Text GitHub](https://github.com/sapientinc/HRM-Text) — cfg_sft, prepare_sft_data.py, flash_attention_prefixlm_v2.py
- [HRM-Text-1B HuggingFace](https://huggingface.co/sapientinc/HRM-Text-1B) — token_type_ids documentation, vLLM "in progress" note
- [sapientinc/data_io](https://github.com/sapientinc/data_io)
- [HRM Perspectives (arXiv 2510.00355)](https://arxiv.org/html/2510.00355v1) — H-module contribution disputed, L-module-only comparable

### Datasets
- [OpenR1-Math-220K](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [OpenR1 intermediate \\boxed{} parsing bug (Discussion #5)](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/discussions/5)
- [DAPO-Math-17K](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) — prompts only, not SFT data
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) — ■ delimiter warning

### RL Methods
- [GRPO: DeepSeekMath (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300) — baseline, G=64, β=0.04
- [DAPO (arXiv 2503.14476)](https://arxiv.org/abs/2503.14476) — ε_high=0.28, β=0.0, long-CoT design
- [REINFORCE++ (arXiv 2501.03262)](https://arxiv.org/abs/2501.03262) — global batch normalization
- [RAFT vs GRPO (arXiv 2504.11343)](https://arxiv.org/abs/2504.11343) — RAFT plateau at 56.1% vs GRPO 56.3%
- [DART-Math RAFT failure (arXiv 2407.13690)](https://arxiv.org/abs/2407.13690) — 51.1% L5 miss rate
- [SDPO (arXiv 2601.20802)](https://arxiv.org/abs/2601.20802) — rejected; needs textual feedback
- [SRPO beats SDPO (arXiv 2604.02288)](https://arxiv.org/html/2604.02288v1) — SDPO 71.1% < GRPO 74.0%
- [Entropy collapse delay (arXiv 2602.03190)](https://arxiv.org/html/2602.03190v1) — DAPO delays, not prevents collapse
- [DAPO clip mechanism (arXiv 2509.26114)](https://arxiv.org/abs/2509.26114) — clip-high/low entropy effects
- [AceReason-Nemotron β=0.0 (arXiv 2506.13284)](https://arxiv.org/html/2506.13284v1)
- [Over-SFT harms RL (arXiv 2510.01624)](https://arxiv.org/abs/2510.01624) — 1 epoch rule
- [ProRL reference resets (arXiv 2505.24864)](https://arxiv.org/abs/2505.24864)
- [LoRA MLP importance (arXiv 2511.06739)](https://arxiv.org/abs/2511.06739) — MLP load-bearing for reasoning
- [Projection LoRA = SSM expressiveness (arXiv 2410.09016)](https://arxiv.org/abs/2410.09016) — ICML 2025
- [LoRA intrinsic dimensionality (arXiv 2012.13255)](https://arxiv.org/abs/2012.13255) — why LoRA works
- [LoRA forgetting (arXiv 2405.09673)](https://arxiv.org/abs/2405.09673) — LoRA < full fine-tuning forgetting
- [General data mixing harm (NVIDIA ADLR)](https://research.nvidia.com/labs/adlr/) — -5% math from naive scaling
- [OpenThoughts quality > quantity (arXiv 2506.04178)](https://arxiv.org/abs/2506.04178) — 1-2 sources best

### Infrastructure
- [Hyak Klone ckpt partition](https://hyak.uw.edu/docs/compute/checkpoint/) — 8-9h GPU walltime, preemption policy
- [Hyak Tillicum architecture](https://hyak.uw.edu/docs/tillicum/architecture/) — H200 specs
- [FA3 hopper install](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing+flash-attention+3+for+hopper)
- [FA3 varlen bug #1570](https://github.com/Dao-AILab/flash-attention/issues/1570)

### Reward
- [HuggingFace Math-Verify](https://github.com/huggingface/Math-Verify)

---

*Research basis: 9 adversarial deep-research passes (~900 agents, ~20M tokens verified). Confidence notation: 3-0 = high (confirmed unanimously), 2-1 = medium (one dissenter), 0-3 = invalidated (unanimously refuted).*
