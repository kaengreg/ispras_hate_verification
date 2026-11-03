#!/usr/bin/env bash
#SBATCH --job-name=simpo_qwen3_32b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=simpo_qwen3_32b_%j.out
#SBATCH --error=simpo_qwen3_32b_%j.err

set -euo pipefail

# Container
WORKDIR="${WORKDIR:-/scratch/pr0204/kaengreg-workdir}"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/ispras}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${WORKDIR}/docker_containers/ispras-simpo-qwen3-32b.sqsh}"

# Model and task
CONFIG="${CONFIG:-${PROJECT_DIR}/training/simpo_training/config_qwen3_32b_qlora.json}"
NPROC_PER_NODE=4
PYTHON_BIN="${PYTHON_BIN:-python}"

# Runtime
export HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORKDIR}/torch-extensions}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ -d "${PROJECT_DIR}" ]] || { echo "Project directory not found: ${PROJECT_DIR}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${CONTAINER_IMAGE}" ]] || { echo "Container image not found: ${CONTAINER_IMAGE}" >&2; exit 1; }
mkdir -p "${HF_HOME}" "${TORCH_EXTENSIONS_DIR}"

srun \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${PROJECT_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        training/simpo_training/simpo_questions.py \
        --config "${CONFIG}"
