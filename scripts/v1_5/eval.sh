OUTPUT_DIR=checkpoints/llava-v1.5-7b-finetune_RobustVLGuard_backbone_lora_4k_2k_3e_STD005-015_P07/merged

cd ~/Project/LLM-Safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
python llava_inference_mmvet.py --model_path /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR} --clean --output_file outputs/mmvet-clean-llava15-7b-lora_4k_2k_3e_STD005-015_P07.json 2>&1 | tee -a "${OUTPUT_DIR}/training_log_clean_mmvet.txt" &
python llava_inference_mmvet.py --model_path /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR} --output_file outputs/mmvet-noisy-llava15-7b-lora_4k_2k_3e_STD005-015_P07.json 2>&1 | tee -a "${OUTPUT_DIR}/training_log_noisy_mmvet.txt" &

wait

bash omi_eval_rtp.sh outputs/defense/clean_llava15-7b-lora_4k_2k_3e_STD005-015_P07/sample adversarial_images/clean.jpeg /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR} 0 2>&1 | tee -a "${OUTPUT_DIR}/training_log_clean_safety.txt" &

bash omi_eval_rtp.sh outputs/defense/noisy_llava15-7b-lora_4k_2k_3e_STD005-015_P07/sample adversarial_images/random_noisy_constrained_16.bmp /home/t-jiaweiwang/Project/LLaVA/${OUTPUT_DIR} 0 2>&1 | tee -a "${OUTPUT_DIR}/training_log_noisy_safety.txt" &

wait
