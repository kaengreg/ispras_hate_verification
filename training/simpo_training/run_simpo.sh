#!/usr/bin/env bash
#SBATCH --job-name=simpo_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=simpo_train_%j.out
#SBATCH --error=simpo_train_%j.err

set -euo pipefail

# Container
WORKDIR="${WORKDIR:-/scratch/pr0204/kaengreg-workdir}"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/ispras}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${WORKDIR}/docker_containers/ispras.sqsh}"

# Training
CONFIG="${CONFIG:-${PROJECT_DIR}/training/simpo_training/config.json}"
NPROC_PER_NODE=4

# Runtime
export HF_HOME="${HF_HOME:-${WORKDIR}/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "${HF_HOME}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${PROJECT_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        training/simpo_training/simpo_questions.py \
        --config "${CONFIG}"
