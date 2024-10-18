#!/bin/bash

export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
export WANDB_RESUME="allow" &&
export WANDB_API_KEY="618eb3b78242f01000855a123d29e2ac98a60f30" &&
export WANDB_PROJECT="compressv" &&
export CKPT_NAME="cambrian_qwen05b_CLIP_mlp_2scale_dim448_layer12_finetune_pad_738k_old_TPU" &&

export CKPT_DIR="gs://cambrian-archive/checkpoints/$CKPT_NAME" &&

python cambrian/train/train_tpu.py \
   --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
    --version qwen_1_5 \
    --data_path ./llava_next_raw_format_processed.jsonl\
    --image_folder ./llava_next \
    --pretrain_mm_mlp_adapter ./cambrian_qwen05b_CLIP_mlp_2scale_dim448_layer12_shareGPT4V_pretrain_pad_old_TPU/mm_projector.bin \
    --vision_tower_aux_list '["openai/clip-vit-large-patch14-336"]' \
    --vision_tower_aux_token_len_list '[576]' \
    --compress_v_start_layer 12 \
    --compress_v True \
    --image_token_len 576 \
    --image_token_len_concise 36 \
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
    --unfreeze_mm_vision_tower False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir $CKPT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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