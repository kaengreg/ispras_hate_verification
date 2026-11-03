#!/usr/bin/env bash
#SBATCH --job-name=simpo_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/pr0204/kaengreg-workdir/ispras/training/logs/simpo_train_%j.out
#SBATCH --error=/scratch/pr0204/kaengreg-workdir/ispras/training/logs/simpo_train_%j.err

set -euo pipefail

# Container and paths
WORKDIR="${WORKDIR:-/scratch/pr0204/kaengreg-workdir}"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/ispras}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/pr0204/dockers/ngc_pytorch_25_03_torch_2.11_vllm_0.21.sqsh}"
CONFIG="${CONFIG:-${PROJECT_DIR}/training/configs/ruadaptqwen3-32b_lr1e-5_e2_r128_a64_beta2.json}"
NPROC_PER_NODE=8

# Runtime
export HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ -d "${PROJECT_DIR}" ]] || { echo "Project directory not found: ${PROJECT_DIR}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${CONTAINER_IMAGE}" ]] || { echo "Container image not found: ${CONTAINER_IMAGE}" >&2; exit 1; }
mkdir -p "${HF_HOME}"

echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
srun --ntasks=1 nvidia-smi -L

srun \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${PROJECT_DIR}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        training/simpo_questions.py \
        --config "${CONFIG}"
