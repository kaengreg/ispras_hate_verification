#!/usr/bin/env bash
#SBATCH --job-name=question_paraphrases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pr0204/kaengreg-workdir/ispras/provocative_bench/logs/question_paraphrases_%j.out
#SBATCH --error=/scratch/pr0204/kaengreg-workdir/ispras/provocative_bench/logs/question_paraphrases_%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/pr0204/kaengreg-workdir}"
BENCH_DIR="${BENCH_DIR:-${WORKDIR}/ispras/provocative_bench}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/pr0204/dockers/ngc_pytorch_25_03_torch_2.11_vllm_0.21.sqsh}"
INPUT="${INPUT:-${BENCH_DIR}/provocative_data/provocative_questions_1k+mid-neg_ready.jsonl}"
OUTPUT="${OUTPUT:-${BENCH_DIR}/provocative_data/provocative_questions_paraphrased.jsonl}"

# Connect to an already running vLLM server; extra script arguments go to Python.
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${BENCH_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    python3 -u "${BENCH_DIR}/provocative_questions/paraphrase_questions.py" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --api-base "${VLLM_BASE_URL:-http://127.0.0.1:8010}" \
    --model "${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}" \
    --workers "${WORKERS:-16}" \
    --resume \
    "$@"
