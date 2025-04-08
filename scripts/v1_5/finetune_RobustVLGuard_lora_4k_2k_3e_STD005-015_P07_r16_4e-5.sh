#!/bin/bash
set -x
export WANDB_API_KEY="d8d48d5c16ca9b51769d812605aed1929aca30e1"
export WANDB_PROJECT="VLM-Safety"
export WANDB_NAME="Comp-4k-Safety-2k_3e_STD005-015_P07-Gaussian-LLaVA-v1.5-7B-Backbone-Lora16-4e-5"

OUTPUT_DIR=checkpoints/llava-v1.5-7b-finetune_RobustVLGuard_backbone_lora_4k_2k_3e_STD005-015_P07_r16_4e-5/

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --use_backbone_lora True --use_llm_lora False --freeze_llm True --freeze_mm_mlp_adapter True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --meta_data_path data_configs/comp4k_safety2k.json \
    --noise_augmentation True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

# python scripts/merge_lora_weights.py --model-path ${OUTPUT_DIR} --model-base liuhaotian/llava-v1.5-7b --save-model-path ${OUTPUT_DIR}/merged

cd ~/Project/LLM-Safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
python llava_inference_mmvet.py --model_path /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/checkpoint-2200 --clean --output_file outputs/mmvet-clean-llava15-7b-lora_4k_2k_3e_STD005-015_P07_4e-5.json 2>&1 | tee -a "/home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/training_log_clean_mmvet.txt" &
python llava_inference_mmvet.py --model_path /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/checkpoint-2200 --output_file outputs/mmvet-noisy-llava15-7b-lora_4k_2k_3e_STD005-015_P07_4e-5.json 2>&1 | tee -a "/home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/training_log_noisy_mmvet.txt" &

wait

bash omi_eval_rtp.sh outputs/defense/clean_llava15-7b-lora_4k_2k_3e_STD005-015_P07_4e-5/sample adversarial_images/clean.jpeg /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/checkpoint-2200 0 2>&1 | tee -a "/home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/training_log_clean_safety.txt" &

bash omi_eval_rtp.sh outputs/defense/noisy_llava15-7b-lora_4k_2k_3e_STD005-015_P07_4e-5/sample adversarial_images/random_noisy_constrained_16.bmp /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/checkpoint-2200 0 2>&1 | tee -a "/home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR}/training_log_noisy_safety.txt" &

wait
