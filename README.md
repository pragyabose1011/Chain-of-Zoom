# Chain-of-Zoom

---
## 🔥 Summary

This repository implements a **Chain-of-Zoom based image understanding pipeline** that progressively refines visual details by iteratively zooming into regions of interest. Instead of processing an image at a single scale, the pipeline performs hierarchical zooming to improve fine-grained visual reasoning and reconstruction.

The project is structured to run **locally end-to-end**, integrating pretrained vision and diffusion components and managing required model checkpoints within the repository. Emphasis is placed on practical execution, reproducibility, and clear organization, making the system easy to run, inspect, and demonstrate.

Overall, this repository serves as an applied implementation of hierarchical zoom-based image processing, showcasing how multi-stage refinement can enhance visual understanding in computer vision workflows.


## 🛠️ Setup
First, create your environment. We recommend using the following commands. 

```
git clone https://github.com/pragyabose1011/Chain-of-Zoom.git
cd Chain-of-Zoom

conda create -n coz python=3.10
conda activate coz
pip install -r requirements.txt
```

## ⚡ Quick Inference
You can quickly check the results of using **CoZ** with the following example:
```
python inference_coz.py \
  -i samples \
  -o inference_results/coz_vlmprompt \
  --rec_type recursive_multiscale \
  --prompt_type vlm \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth \
  --save_prompts;
```
Which will give a result like below:

![main figure](assets/example_result.png)


## 🌄 Full Image Super-Resolution
Although our main focus is zooming into local areas, **CoZ** can be easily applied to super-resolution of full images. Try out the code below!

```
python inference_coz_full.py \
  -i samples \
  -o inference_results/coz_full \
  --rec_type recursive_multiscale \
  --prompt_type vlm \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth;
```

## 🚆 Training the SR Backbone Model
**Chain-of-Zoom** is model-agnostic and can be used with *any* pretrained text-aware SR model. In this repository we leverage OSEDiff trained with Stable Diffusion 3 Medium as its backbone model. This requires some additional installations:

```
pip install wandb opencv-python basicsr==1.4.2

pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 xformers==0.0.28.post1
```

Please refer to the [OSEDiff](https://github.com/cswry/OSEDiff) repository for training configurations (ex. preparing training data). Now train the SR backbone model:
```
bash scripts/train/train_osediff_sd3.sh
```


