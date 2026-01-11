#----------------- Full Image CoZ with VLM Prompts -----------------#
# Uses a GRPO trained VLM LoRA Adapter
# REQUIRED ENVIRONMENT: coz

INPUT_FOLDER="samples"
OUTPUT_FOLDER="inference_results/coz_full"

CUDA_VISIBLE_DEVICES=0,1, python inference_coz_full.py \
-i $INPUT_FOLDER \
-o $OUTPUT_FOLDER \
--rec_type recursive_multiscale \
--prompt_type vlm \
--lora_path ckpt/SR_LoRA/model_20001.pkl \
--vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
--vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
--pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
--ram_ft_path ckpt/DAPE/DAPE.pth \
--ram_path ckpt/RAM/ram_swin_large_14m.pth;

