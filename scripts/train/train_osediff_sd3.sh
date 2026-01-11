# Training OSEDiff with Stable Diffusion 3.0 Backbone
# Requires 4 * 24GB GPUs

CUDA_VISIBLE_DEVICES="0,1,2,3," python train_osediff_sd3.py \
    --pretrained_model_name_or_path='stabilityai/stable-diffusion-3-medium-diffusers' \
    --ram_path='ckpt/RAM/ram_swin_large_14m.pth' \
    --learning_rate=5e-5 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps 5000 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --output_dir=experience/osediff_sd3 \
    --dataset_txt_paths_list './train_utils/dataset_paths/LSDIR_TRAIN.txt','./train_utils/dataset_paths/FFHQ512_TRAIN10K.txt' \
    --dataset_prob_paths_list 1,1 \
    --neg_prompt="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth" \
    --cfg_vsd=7.5 \
    --lora_rank=4 \
    --lambda_lpips=2 \
    --lambda_l2=1 \
    --lambda_vsd=1 \
    --lambda_vsd_lora=1 \
    --deg_file_path="params_realesrgan.yml" \
    --tracker_project_name "train_osediff" \
    --gradient_checkpointing \
    --log_wandb;