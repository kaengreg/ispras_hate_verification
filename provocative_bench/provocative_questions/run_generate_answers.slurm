#!/usr/bin/env bash
#SBATCH --job-name=provocative_answers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pr0204/kaengreg-workdir/ispras/provocative_bench/logs/provocative_answers_%j.out
#SBATCH --error=/scratch/pr0204/kaengreg-workdir/ispras/provocative_bench/logs/provocative_answers_%j.err

set -euo pipefail

# Paths
WORKDIR="${WORKDIR:-/scratch/pr0204/kaengreg-workdir}"
BENCH_DIR="${BENCH_DIR:-${WORKDIR}/ispras/provocative_bench}"
INPUT="${INPUT:-${BENCH_DIR}/provocative_data/provocative_questions_paraphrased.jsonl}"
OUTPUT="${OUTPUT:-${BENCH_DIR}/provocative_data/provocative_answers.jsonl}"

# Container and model endpoint
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/pr0204/dockers/ngc_pytorch_25_03_torch_2.11_vllm_0.21.sqsh}"
MODEL="${MODEL:?Set MODEL to the served vLLM model name}"
API_BASE="${VLLM_BASE_URL:-http://127.0.0.1:8000}"

# Generation
WORKERS="${WORKERS:-32}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# Connect to an already running OpenAI-compatible vLLM server.
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${BENCH_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    python3 -u "${BENCH_DIR}/generate_answers_vllm.py" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --api-base "${API_BASE}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --resume \
    "$@"
