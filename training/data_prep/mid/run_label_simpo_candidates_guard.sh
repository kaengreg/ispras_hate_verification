#!/usr/bin/env bash
#SBATCH --job-name=simpo_guard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=label_simpo_guard_%j.out
#SBATCH --error=label_simpo_guard_%j.err

set -euo pipefail

# Container and paths
WORKDIR="/scratch/pr0204/kaengreg-workdir"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/ispras}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${WORKDIR}/docker_containers/ispras.sqsh}"
INPUT="${INPUT:-${PROJECT_DIR}/training/data/mid/simpo_candidates/simpo_candidates_limited_filtered.jsonl}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/training/data/mid/simpo_candidates/simpo_candidates_limited_filtered_guard.jsonl}"

# Guard model and vLLM endpoint
VLLM_MODEL="${VLLM_MODEL:-RefalMachine/RuadaptQwen3-32B-Instruct}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
WORKERS="${WORKERS:-32}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${PROJECT_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    python training/data_prep/mid/label_simpo_candidates_guard.py \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        --model "${VLLM_MODEL}" \
        --api-base "${VLLM_BASE_URL}" \
        --workers "${WORKERS}" \
        --with-question-context \
        --resume
