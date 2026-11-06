#!/usr/bin/env bash
#SBATCH --job-name=provocative_questions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=provocative_questions_%j.out
#SBATCH --error=provocative_questions_%j.err

set -euo pipefail

# Container and paths
WORKDIR="${WORKDIR:-/scratch/pr0204/kaengreg-workdir}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/ispras}"
BENCH_DIR="${BENCH_DIR:-${REPO_DIR}/provocative_bench}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/pr0204/dockers/ngc_pytorch_25_03_torch_2.11_vllm_0.21.sqsh}"
INPUT="${INPUT:-${BENCH_DIR}/provocative_data/provocative_question_sources_all.jsonl}"
OUTPUT="${OUTPUT:-${BENCH_DIR}/provocative_data/provocative_questions_all.jsonl}"
RESUME_FROM="${RESUME_FROM:-${BENCH_DIR}/provocative_data/provocative_questions_2000.jsonl}"

# Model and generation
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
PORT="${PORT:-8010}"
WORKERS="${WORKERS:-32}"
VLLM_LOG="${VLLM_LOG:-${BENCH_DIR}/vllm_questions_${SLURM_JOB_ID}.log}"

# Runtime
export HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export BENCH_DIR INPUT OUTPUT RESUME_FROM MODEL PORT WORKERS VLLM_LOG
export QREVIEW_MODEL="${MODEL}"
export QREVIEW_SERVED_NAME="${MODEL}"
export QREVIEW_PORT="${PORT}"
export QREVIEW_GPU_IDS="0,1,2,3"
export QREVIEW_TENSOR_PARALLEL_SIZE="4"

[[ -d "${BENCH_DIR}" ]] || { echo "Benchmark directory not found: ${BENCH_DIR}" >&2; exit 1; }
[[ -f "${INPUT}" ]] || { echo "Input not found: ${INPUT}" >&2; exit 1; }
[[ -f "${CONTAINER_IMAGE}" ]] || { echo "Container image not found: ${CONTAINER_IMAGE}" >&2; exit 1; }
mkdir -p "${HF_HOME}" "$(dirname "${OUTPUT}")"

srun \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${BENCH_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    bash -lc '
        set -euo pipefail

        cleanup() {
            if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
                kill "${VLLM_PID}" 2>/dev/null || true
                wait "${VLLM_PID}" 2>/dev/null || true
            fi
        }
        trap cleanup EXIT INT TERM

        provocative_questions/start_qwen122b_vllm.sh >"${VLLM_LOG}" 2>&1 &
        VLLM_PID=$!

        ready=0
        for _ in $(seq 1 360); do
            if python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:${PORT}/v1/models\", timeout=2).read()" 2>/dev/null; then
                ready=1
                break
            fi
            if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                echo "vLLM stopped before becoming ready" >&2
                tail -n 100 "${VLLM_LOG}" >&2 || true
                exit 1
            fi
            sleep 5
        done

        if [[ "${ready}" != 1 ]]; then
            echo "vLLM did not become ready within 30 minutes" >&2
            tail -n 100 "${VLLM_LOG}" >&2 || true
            exit 1
        fi

        resume_from_args=()
        if [[ -f "${RESUME_FROM}" ]]; then
            resume_from_args=(--resume-from "${RESUME_FROM}")
        fi

        python provocative_questions/generate_questions_qwen122b.py \
            --input "${INPUT}" \
            --output "${OUTPUT}" \
            --api-base "http://127.0.0.1:${PORT}" \
            --model "${MODEL}" \
            --workers "${WORKERS}" \
            "${resume_from_args[@]}" \
            --resume
    '
