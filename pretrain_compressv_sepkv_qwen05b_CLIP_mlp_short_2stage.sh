#!/bin/bash

export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
export WANDB_RESUME="allow" &&
export WANDB_API_KEY="618eb3b78242f01000855a123d29e2ac98a60f30" &&
export WANDB_PROJECT="compressv" &&
export CKPT_NAME="compressv_qwen05b_CLIP_mlp_sepsa_nosvd_oproj448_layer0_shareGPT4V_square_pretrain_stage2joint_TPU" &&

export CKPT_DIR="gs://cambrian-archive/checkpoints/$CKPT_NAME" &&

python cambrian/train/train_tpu.py \
    --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
    --version qwen_1_5 \
    --data_path /mnt/disks/storage/data/finetune_data/pretrain.jsonl \
    --image_folder /mnt/disks/storage/data/finetune_data \
    --vision_tower_aux_list '["openai/clip-vit-large-patch14-336"]' \
    --vision_tower_aux_token_len_list '[576]' \
    --pretrain_mm_mlp_adapter ./compressv_qwen05b_CLIP_mlp_baseline_shareGPT4V_square_pretrain_TPU/mm_projector.bin \
    --max_num_image_crops 1 \
    --per_crop_token_len 576 \
    --compress_reduce_factor 4 \
    --compress_v True \
    --compress_v_start_layer 0 \
    --mm_vision_mlp_lr 1e-4 \
    --num_query_group 1 \
    --query_num_list '[576]' \
    --connector_depth 3 \
    --image_position 14 \
    --vision_hidden_size 1024 \
    --connector_only True \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --image_aspect_ratio square \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir $CKPT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $CKPT_NAME \
    --fsdp "full_shard" \
    --fsdp_config fsdp_config.json


# CKPT_PATH=checkpoints/$CKPT_NAME
# # check if the checkpoint path exists
# if [ ! -d "$CKPT_PATH" ]; then
#     echo "Checkpoint path does not exist. Exiting..."
#     exit 1
# fi
# echo "Training finished. Syncing checkpoints to GCS..."
# gcloud alpha storage rsync $CKPT_PATH gs://cambrian-archive/checkpoints/$CKPT_NAME
# echo "Syncing finished. Checkpoints are now available at gs://cambrian-archive/checkpoints/$CKPT_NAME"
