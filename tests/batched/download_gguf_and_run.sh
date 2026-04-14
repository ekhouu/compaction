#!/usr/bin/env bash
set -euo pipefail

# Auto-download a GGUF model, then run batched benchmarks.
#
# Defaults can be overridden via env vars:
#   GGUF_REPO=Qwen/Qwen3-4B-GGUF
#   GGUF_FILE=Qwen3-4B-Q4_K_M.gguf
#   GGUF_DIR=/home/mikhoiuo/repos/compaction/.gguf-models
#   UV_CACHE_DIR=/tmp/uv-cache
#
# Optional flags:
#   --download-only   Download (or verify existing) then exit.
#   --skip-download   Skip download step and require local file.
#
# Any remaining args are passed to tests/batched/run_batched_benchmarks.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GGUF_REPO="${GGUF_REPO:-Qwen/Qwen3-4B-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen3-4B-Q4_K_M.gguf}"
GGUF_DIR="${GGUF_DIR:-${REPO_ROOT}/.gguf-models}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

DOWNLOAD_ONLY=0
SKIP_DOWNLOAD=0

RUNNER_ARGS=()
while (($#)); do
  case "$1" in
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --gguf-model)
      echo "Do not pass --gguf-model to this wrapper; it is set automatically." >&2
      exit 2
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "${GGUF_DIR}"
GGUF_PATH="${GGUF_DIR}/${GGUF_FILE}"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  if [[ -f "${GGUF_PATH}" ]]; then
    echo "GGUF already present: ${GGUF_PATH}"
  else
    if ! command -v hf >/dev/null 2>&1; then
      echo "Missing 'hf' CLI. Install it first: https://huggingface.co/docs/huggingface_hub/en/guides/cli" >&2
      exit 1
    fi
    echo "Downloading ${GGUF_REPO}/${GGUF_FILE} -> ${GGUF_DIR}"
    hf download "${GGUF_REPO}" "${GGUF_FILE}" --local-dir "${GGUF_DIR}"
  fi
fi

if [[ ! -f "${GGUF_PATH}" ]]; then
  echo "GGUF file not found: ${GGUF_PATH}" >&2
  exit 1
fi

if [[ "${DOWNLOAD_ONLY}" == "1" ]]; then
  echo "Download complete: ${GGUF_PATH}"
  exit 0
fi

if [[ ${#RUNNER_ARGS[@]} -eq 0 ]]; then
  echo "No benchmark args supplied. Example:" >&2
  echo "  $0 --mode all --all-order tq,compaction,both --both-strategy parallel --llama-dir /path/to/llama.cpp" >&2
  exit 2
fi

cd "${REPO_ROOT}"
UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python tests/batched/run_batched_benchmarks.py \
  --gguf-model "${GGUF_PATH}" \
  "${RUNNER_ARGS[@]}"
