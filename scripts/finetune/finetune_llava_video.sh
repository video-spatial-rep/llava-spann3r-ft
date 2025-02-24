#!/bin/bash

# Function to print logs with timestamp
log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}


TIMESTAMP=$(TZ="America/New_York" date "+%Y%m%d_%H%M%S")
log "Starting at $TIMESTAMP"

# setting for single node
export NNODES=1
export RANK=0
export ADDR="localhost"
# export PORT=29500
PORT=$(( ( RANDOM % 64512 ) + 1024 ))
log "using $ADDR:$PORT for master_addr:master_port"


NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export NUM_GPUS


log "NUM_GPUS: ${NUM_GPUS}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp226s0f0
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 
LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"


############### Show Envs ####################

nvidia-smi


############### Pretrain ################

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# log "BASE_RUN_NAME: ${BASE_RUN_NAME}"



############### Params ################
# defaults
LR=1e-5
VIS_LR=2e-6

# half lr
# LR=5e-6
# VIS_LR=1e-6

# FRAMES=64
FRAMES=32

# PD_BS=1
# PD_BS=2  # L40
PD_BS=1  # H100
# PD_BS=8

# GA_STEPS=2
GA_STEPS=4
log "effective batch size: $((PD_BS * GA_STEPS * NUM_GPUS))
    - accumulation steps: $GA_STEPS
    - per device batch size: $PD_BS"
EPOCHS=1

############### Configure Data ################

# folders3
DATA_YAML_PATH="/home/rilyn/scripts/LLaVA-Video-FT/scripts/finetune/vsibench_finetune_data.yaml"
# IMAGE_FOLDER=""
VIDEO_FOLDER="/nas/spatial/source_datasets/scannet/datasets/scans_videos"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
NOTE="_default_vsibench_testrun"
RUN_NAME="ft-llava-video-${LR}-all-0.05-scannet_only-cut3r-${LLM_VERSION_CLEAN}_${TIMESTAMP}"

# PREV_STAGE_CHECKPOINT="/data_new/spatial/huggingface/hub/models--lmms-lab--LLaVA-Video-7B-Qwen2"
PREV_STAGE_CHECKPOINT="/nas/spatial/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-0.5b-ov/snapshots/381d9947148efb1e58a577f451c05705ceec666e"

OUTPUT_CHECKPOINT="/data/rilyn/checkpoints/$RUN_NAME"

log "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
log "MID_RUN_NAME: ${RUN_NAME}"


############### Configure WandB ################

export WANDB_PROJECT=llava_ft_0.5
export WANDB_NAME=$RUN_NAME
# export WANDB_MODE=offline
############### Run ################

# removed     --image_folder $IMAGE_FOLDER \

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
deepspeed --master_port 30000 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML_PATH \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=$VIS_LR \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_CHECKPOINT \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $PD_BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GA_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound $FRAMES \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2

log "Finished at $(TZ="America/New_York" date "+%Y%m%d_%H%M%S")"
