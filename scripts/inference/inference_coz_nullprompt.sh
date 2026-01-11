#----------------- CoZ with Null Prompts -----------------#
# REQUIRED ENVIRONMENT: coz

INPUT_FOLDER="samples"
OUTPUT_FOLDER="inference_results/coz_nullprompt"

CUDA_VISIBLE_DEVICES=0,1, python inference_coz.py \
-i $INPUT_FOLDER \
-o $OUTPUT_FOLDER \
--rec_type recursive \
--prompt_type null \
--lora_path ckpt/SR_LoRA/model_20001.pkl \
--vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
--pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers';

