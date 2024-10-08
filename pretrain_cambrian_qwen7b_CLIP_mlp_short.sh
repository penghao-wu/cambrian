#!/bin/bash

export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
export WANDB_RESUME="allow" &&
export CKPT_NAME="cambrian_qwen05b_CLIP_mlp_576_shareGPT4V_pretrain_square_plain" &&

export CKPT_DIR="gs://cambrian-archive/checkpoints/$CKPT_NAME" &&
export WANDB_API_KEY="618eb3b78242f01000855a123d29e2ac98a60f30" &&
export WANDB_PROJECT="compressv" &&
python cambrian/train/train_tpu.py \
    --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
    --version qwen_1_5 \
    --data_path /mnt/disks/storage/data/pretrain_data/sbu558k/blip_laion_cc_sbu_558k.jsonl \
    --image_folder /mnt/disks/storage/data/pretrain_data/sbu558k \
    --vision_tower_aux_list '["openai/clip-vit-large-patch14-336"]' \
    --vision_tower_aux_token_len_list '[576]' \
    --image_token_len 576 \
    --num_query_group 1 \
    --query_num_list '[576]' \
    --connector_depth 3 \
    --image_position 14 \
    --vision_hidden_size 1024 \
    --connector_only True \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_sampler_lr 1e-3 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --bf16 False \
    --output_dir $CKPT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
