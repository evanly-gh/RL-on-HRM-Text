#!/bin/bash
#SBATCH --job-name=hrm-sft
#SBATCH --partition=gpu-h200
#SBATCH --account=intelligentsystems-ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180

# ── Paths — set these before submitting ──────────────────────────────────────
# PRETRAIN_CKPT: path to the downloaded HRM-Text-1B pretrained checkpoint
# SFT_DATA:      path to the prepared (tokenized) SFT data directory
# SFT_OUT:       where to write the SFT checkpoint
PRETRAIN_CKPT=${PRETRAIN_CKPT:-/mmfs1/gscratch/intelligentsystems/$USER/checkpoints/hrm-text-1b}
SFT_DATA=${SFT_DATA:-/mmfs1/gscratch/intelligentsystems/$USER/sft-data/sft_tokenized}
SFT_OUT=${SFT_OUT:-/mmfs1/gscratch/intelligentsystems/$USER/sft-runs/sft-math-v1}

REPO_ROOT=${REPO_ROOT:-/mmfs1/gscratch/intelligentsystems/$USER/HRM-Text}
ENV_PATH=${ENV_PATH:-/mmfs1/gscratch/intelligentsystems/$USER/hrm_env}

# ── Env ───────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p "${SFT_OUT}" logs

export PATH="${ENV_PATH}/bin:${PATH}"
export HF_HOME=${HF_HOME:-/mmfs1/gscratch/intelligentsystems/$USER/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

echo "[sft] job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi -L | head -1)"

# ── Wall-clock budget ─────────────────────────────────────────────────────────
MARGIN=900
MAX_RUNTIME=0
END_RAW=$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null \
          | grep -oE 'EndTime=[^ ]+' | cut -d= -f2 || true)
if [[ -n "${END_RAW}" && "${END_RAW}" != "Unknown" ]]; then
  END_EPOCH=$(date -d "${END_RAW}" +%s 2>/dev/null || echo 0)
  NOW_EPOCH=$(date +%s)
  if [[ "${END_EPOCH}" -gt "${NOW_EPOCH}" ]]; then
    MAX_RUNTIME=$(( END_EPOCH - NOW_EPOCH - MARGIN ))
    [[ "${MAX_RUNTIME}" -lt 60 ]] && MAX_RUNTIME=60
  fi
fi
echo "[sft] wall-clock budget: MAX_RUNTIME=${MAX_RUNTIME}s"

# ── Resume from latest checkpoint if available ────────────────────────────────
LATEST=$(ls -td "${SFT_OUT}"/checkpoint-* 2>/dev/null | head -n1 || true)
RESUME_ARGS=""
if [[ -n "${LATEST}" ]]; then
  echo "[sft] resuming from ${LATEST}"
  RESUME_ARGS="resume_from=${LATEST}"
else
  echo "[sft] starting fresh from pretrain checkpoint: ${PRETRAIN_CKPT}"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# Single H200, single process (no FSDP2 overhead at 1B scale)
cd "${REPO_ROOT}"

torchrun --standalone --nproc_per_node=1 pretrain.py \
  --config-name cfg_sft \
  "data.path=${SFT_DATA}" \
  "resume_from=${PRETRAIN_CKPT}" \
  "+checkpoint_path=${SFT_OUT}" \
  "weights_only_resume_from_ema=true" \
  "epochs=1" \
  "lr=2e-5" \
  "global_batch_size=32768" \
  "max_runtime_seconds=${MAX_RUNTIME}" \
  ${RESUME_ARGS} \
  > >(tee -a "${SFT_OUT}/train.log") 2>&1 &

TRAIN_PID=$!
trap 'echo "[sft] USR1 -> forwarding to trainer"; kill -USR1 "${TRAIN_PID}" 2>/dev/null || true' USR1

RC=""
while [[ -z "${RC}" ]]; do
  if wait "${TRAIN_PID}"; then RC=0
  else
    code=$?
    kill -0 "${TRAIN_PID}" 2>/dev/null && continue
    RC=${code}
  fi
done
echo "[sft] trainer exit=${RC}"

# ── Self-resubmit if preempted ────────────────────────────────────────────────
RESUBMIT_MAX=${SFT_RESUBMIT_MAX:-10}
RESUBMIT_N=${SFT_RESUBMIT_N:-0}

if [[ -f "${SFT_OUT}/COMPLETE" ]]; then
  echo "[sft] Training COMPLETE — not resubmitting."
elif [[ "${RC}" -eq 0 ]] && ls -d "${SFT_OUT}"/checkpoint-* >/dev/null 2>&1; then
  if [[ "${RESUBMIT_N}" -ge "${RESUBMIT_MAX}" ]]; then
    echo "[sft] hit SFT_RESUBMIT_MAX=${RESUBMIT_MAX}; stopping chain."
  else
    echo "[sft] resubmitting (chain ${RESUBMIT_N} -> $((RESUBMIT_N+1)))"
    sbatch --export=ALL,SFT_RESUBMIT_N=$((RESUBMIT_N+1)),\
PRETRAIN_CKPT="${PRETRAIN_CKPT}",SFT_DATA="${SFT_DATA}",\
SFT_OUT="${SFT_OUT}",REPO_ROOT="${REPO_ROOT}",ENV_PATH="${ENV_PATH}" \
      "${REPO_ROOT}/sft-data/submit_sft.sbatch"
  fi
else
  echo "[sft] exit=${RC} with no checkpoint — check logs."
fi

exit "${RC}"
