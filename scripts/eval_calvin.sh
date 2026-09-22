#!/usr/bin/env bash
# CALVIN inference / evaluation launcher.
#
# Usage:
#   bash scripts/eval_calvin.sh                          # full eval on the default checkpoint below
#   bash scripts/eval_calvin.sh /path/to/model_XXXX.pth  # full eval on a specific checkpoint
#   SMOKE=1 bash scripts/eval_calvin.sh                  # quick smoke test (10 seqs, no videos)
#   NPROC=1 bash scripts/eval_calvin.sh                  # override number of processes
#
# Notes (learned the hard way):
#   * Run SINGLE process (NPROC=1). Launching one process per GPU makes each
#     spin up its own pybullet+EGL context and they deadlock in loadURDF during
#     CALVIN env creation. One process avoids the contention.
#   * The checkpoint is a full-model state_dict, loaded via weights.model. Uses
#     `++` (upsert) for all three weights.* keys since infer.yaml's `weights` struct
#     only declares `model` today; `++` works whether or not a key is declared,
#     so this doesn't break again if infer.yaml's struct changes.
#     motion_director/action_expert are nulled so the stale defaults in infer.yaml
#     don't overwrite the fresh weights.
#   * DATASET points at the CALVIN validation set that has .hydra/merged_config.yaml.
#   * wandb logging is disabled (accelerator.log_with=null): wandb's background
#     service process has an internal asyncio assertion bug that reliably crashes
#     the run a few sequences in (AssertionError in asyncio/streams.py _drain_helper
#     during accelerator.log()). Rollout evaluation itself works fine; only the
#     wandb tracker call was failing.
set -euo pipefail

# ---- config (override via env or first arg) --------------------------------
CKPT="${1:-/home/colligo/Codes/localssd/DAWN/2026-09-12_18-24/checkpoints/model_0041000.pth}"
DATASET="${DATASET:-/mnt/localssd/calvin/validation}"
NPROC="${NPROC:-1}"
PORT="${PORT:-19500}"

# ---- smoke-test toggle ------------------------------------------------------
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA+=(inference.num_sequences=10 inference.record_rollout_videos=False inference.record_flow=False)
fi

mkdir -p logs

echo "Checkpoint : ${CKPT}"
echo "Dataset    : ${DATASET}"
echo "Processes  : ${NPROC}"
echo "Extra args : ${EXTRA[*]:-<none>}"

accelerate launch --num_processes="${NPROC}" --main_process_port="${PORT}" inference.py \
  ++weights.model="${CKPT}" \
  ++weights.motion_director=null \
  ++weights.action_expert=null \
  ++accelerator.log_with=null \
  inference.dataset="${DATASET}" \
  "${EXTRA[@]}"
