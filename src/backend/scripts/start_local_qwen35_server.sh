#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VENV_DIR="${EM2MEM_LOCAL_LLM_VENV:-$ROOT_DIR/.venv_vllm}"
MODEL_PATH="${EM2MEM_LOCAL_LLM_MODEL_PATH:-$ROOT_DIR/models/Qwen3.5-9B}"
HOST="${EM2MEM_LOCAL_LLM_HOST:-127.0.0.1}"
PORT="${EM2MEM_LOCAL_LLM_PORT:-18100}"
SERVED_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL:-gpt-5.4}"
GPU="${EM2MEM_LOCAL_LLM_CUDA_VISIBLE_DEVICES:-2}"
MAX_MODEL_LEN="${EM2MEM_LOCAL_LLM_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${EM2MEM_LOCAL_LLM_GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${EM2MEM_LOCAL_LLM_MAX_NUM_SEQS:-2}"
TMP_DIR="${EM2MEM_LOCAL_LLM_TMPDIR:-$ROOT_DIR/.cache/vllm-tmp}"
PYTHON_SITE="$($VENV_DIR/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDA_HOME_DEFAULT="$PYTHON_SITE/nvidia/cu13"

if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "[start_local_qwen35_server] missing vLLM environment; run scripts/setup_local_qwen35_env.sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[start_local_qwen35_server] missing model at $MODEL_PATH; run scripts/download_local_qwen35_model.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME="${EM2MEM_LOCAL_LLM_CUDA_HOME:-$CUDA_HOME_DEFAULT}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$VENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$ROOT_DIR}"
mkdir -p "$TMP_DIR"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
export TMPDIR="$TMP_DIR"

echo "[start_local_qwen35_server] model=$MODEL_PATH served_model=$SERVED_MODEL gpu=$GPU"
echo "[start_local_qwen35_server] endpoint=http://$HOST:$PORT/v1 max_model_len=$MAX_MODEL_LEN"
echo "[start_local_qwen35_server] CUDA_HOME=$CUDA_HOME"

exec "$VENV_DIR/bin/vllm" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --reasoning-parser qwen3 \
  --trust-remote-code
