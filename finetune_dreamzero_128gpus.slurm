#!/bin/bash
#SBATCH -A healthcareeng_bionemo
#SBATCH --partition=batch
#SBATCH --exclude=pool0-01778
#SBATCH --exclude=pool0-01780
#SBATCH --exclude=pool0-01626
#SBATCH --exclude=pool0-01307
#SBATCH --exclude=pool0-01574
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --ntasks=16
#SBATCH --time=4:00:00
#SBATCH -o /lustre/fsw/portfolios/healthcareeng/users/haochens/dreamzero_logs/test-%j.out
#SBATCH -e /lustre/fsw/portfolios/healthcareeng/users/haochens/dreamzero_logs/test-%j.err

# ============================================
# Auto-resubmission configuration
# ============================================
MAX_RUNS=40
RUN_COUNT=${RUN_COUNT:-1}

echo "=== DreamZero Finetune Run ${RUN_COUNT}/${MAX_RUNS} (Job ID: ${SLURM_JOB_ID}) ==="

CONTAINER_IMAGE="gitlab-master.nvidia.com/holoscan/i4h-containers:dreamzero_unitree_g1"

BASE_PATH="/lustre/fsw/portfolios/healthcareeng/users/haochens"

MOUNTS="${BASE_PATH}/dreamzero_unitree_g1:/workspace"
MOUNTS="${MOUNTS},${BASE_PATH}/data/unitree_dataset/gear:/data"
MOUNTS="${MOUNTS},${BASE_PATH}/dreamzero_ckpts:/workspace/checkpoints"
MOUNTS="${MOUNTS},${BASE_PATH}/cache:/root/.cache"
MOUNTS="${MOUNTS},/tmp:/tmp"

RUN_NAME="assemble_trocar_gear"
DATA_ROOT="/data/${RUN_NAME}"
OUTPUT_DIR="/workspace/checkpoints/${RUN_NAME}"
WAN_CKPT_DIR="/workspace/checkpoints/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="/workspace/checkpoints/umt5-xxl"
PRETRAINED_MODEL_PATH="/workspace/checkpoints/DreamZero-AgiBot"

HOST_OUTPUT_DIR="${BASE_PATH}/dreamzero_ckpts/${RUN_NAME}"

export DATA_ROOT RUN_NAME OUTPUT_DIR WAN_CKPT_DIR TOKENIZER_DIR PRETRAINED_MODEL_PATH HOST_OUTPUT_DIR

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-0} % 20000)))
export MASTER_ADDR MASTER_PORT

timeout 3.9h srun --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace \
    --export=ALL,HF_HOME=/root/.cache/huggingface,TRITON_CACHE_DIR=/tmp/triton,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,HYDRA_FULL_ERROR=1,PYTHONPATH=/workspace \
    bash -c '

mkdir -p /tmp/triton

pip install -e /workspace --quiet --no-deps 2>/dev/null || true

python3 -c "import torch; import flash_attn; print(f\"torch {torch.__version__} flash_attn {flash_attn.__version__} OK\")"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
NODE_RANK="${SLURM_PROCID:-0}"

torchrun \
    --nnodes="${SLURM_NNODES:-16}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/unitree_g1_upper_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir="${OUTPUT_DIR}" \
    per_device_train_batch_size=1 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=4 \
    image_resolution_width=320 \
    image_resolution_height=240 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=1200 \
    save_strategy=steps \
    unitree_g1_upper_data_root="${DATA_ROOT}" \
    dit_version="${WAN_CKPT_DIR}" \
    text_encoder_pretrained_path="${WAN_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="${WAN_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="${WAN_CKPT_DIR}/Wan2.1_VAE.pth" \
    tokenizer_path="${TOKENIZER_DIR}" \
    pretrained_model_path="${PRETRAINED_MODEL_PATH}" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
'

EXIT_CODE=$?

# ============================================
# Auto-resubmit if timeout (exit code 124) and runs remaining
# ============================================
if [[ $EXIT_CODE == 124 ]] && [[ $RUN_COUNT -lt $MAX_RUNS ]]; then
    if [[ -d "${HOST_OUTPUT_DIR}" ]]; then
        INCOMPLETE_DIR="${HOST_OUTPUT_DIR}/_incomplete_checkpoints"
        mkdir -p "${INCOMPLETE_DIR}"
        for ckpt in "${HOST_OUTPUT_DIR}"/checkpoint-*; do
            [[ -d "${ckpt}" ]] || continue
            if [[ ! -s "${ckpt}/trainer_state.json" ]]; then
                base="$(basename "${ckpt}")"
                dest="${INCOMPLETE_DIR}/${base}.$(date +%s)"
                echo "=== Quarantining incomplete checkpoint: ${ckpt} -> ${dest} ==="
                mv "${ckpt}" "${dest}" || true
            fi
        done
    fi

    NEXT_RUN=$((RUN_COUNT + 1))
    echo "=== Timeout reached. Resubmitting as run ${NEXT_RUN}/${MAX_RUNS} ==="
    sbatch --export=ALL,RUN_COUNT=${NEXT_RUN} "$0"
elif [[ $EXIT_CODE == 0 ]]; then
    echo "=== Training completed successfully ==="
else
    echo "=== Training failed with exit code ${EXIT_CODE} ==="
fi
