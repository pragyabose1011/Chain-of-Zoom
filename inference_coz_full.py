import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from peft import PeftModel

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--process_size', type=int, default=512)
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--align_method', type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
    parser.add_argument('--lora_path', type=str, default=None, help='for LoRA of SR model')
    parser.add_argument('--vae_path', type=str, default=None)
    parser.add_argument('--vlm_lora_path', type=str, default=None, help='Path to the VLM LoRA adapter directory')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--prompt_type', type=str, choices=['vlm_base','vlm'], default='vlm', help='type of prompt to use')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--merge_and_unload_lora', action='store_true', help='merge lora weights before inference')
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--rec_type', type=str, choices=['recursive_multiscale'], default='recursive_multiscale', help='type of inference to use')
    parser.add_argument('--rec_num', type=int, default=1)
    
    parser.add_argument('--vae_encoder_tiled_size', type=int, default=1024)
    parser.add_argument('--vae_decoder_tiled_size', type=int, default=128)
    parser.add_argument('--latent_tiled_size', type=int, default=64)
    parser.add_argument('--latent_tiled_overlap', type=int, default=16)
    
    parser.add_argument('--save_prompts', default=False, action='store_true')
    args = parser.parse_args()

    global weight_dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # initialize SR model
    model = None
    from osediff_sd3 import OSEDiff_SD3_TEST_TILE, SD3Euler
    model = SD3Euler()
    model.text_enc_1.to('cuda:0')
    model.text_enc_2.to('cuda:0')
    model.text_enc_3.to('cuda:0')
    model.transformer.to('cuda:1', dtype=torch.float32)
    model.vae.to('cuda:1', dtype=torch.float32)
    for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
        p.requires_grad_(False)
    model_test = OSEDiff_SD3_TEST_TILE(args, model)

    # gather input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # load DAPE if needed
    DAPE = None
    if args.prompt_type == "dape":
        DAPE = ram(pretrained=args.ram_path,
                   pretrained_condition=args.ram_ft_path,
                   image_size=384,
                   vit='swin_l')
        DAPE.eval().to("cuda")
        DAPE = DAPE.to(dtype=weight_dtype)

    # load VLM pipeline if needed
    vlm_model = None
    global vlm_processor
    global process_vision_info
    vlm_processor = None
    if args.prompt_type in ('vlm','vlm_base'):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        vlm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Loading base VLM model: {vlm_model_name}")
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
        print('Base VLM LOADING COMPLETE')
        
        if args.prompt_type == "vlm":
            if not args.vlm_lora_path:
                raise ValueError("Please specify --vlm_lora_path when using prompt_type 'vlm'")
            if not os.path.isdir(args.vlm_lora_path):
                raise ValueError(f"VLM LoRA path does not exist or is not a directory: {args.vlm_lora_path}")

            # load the GRPO fine-tuned VLM LoRA adapter
            print(f"Loading VLM LoRA adapter from: {args.vlm_lora_path}")
            vlm_model = PeftModel.from_pretrained(vlm_model, args.vlm_lora_path)
            vlm_model = vlm_model.merge_and_unload()
            vlm_model.eval()
            print('VLM LoRA ADAPTER LOADING COMPLETE')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-sample'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-scale'), exist_ok=True)
    print(f'There are {len(image_names)} images.')
    print(f'Align Method Used: {args.align_method}')
    print(f'Prompt Type: {args.prompt_type}')

    # inference loop
    for image_name in image_names:
        bname = os.path.basename(image_name)
        rec_dir = os.path.join(args.output_dir, 'per-sample', bname[:-4])
        os.makedirs(rec_dir, exist_ok=True)
        if args.save_prompts:
            txt_path = os.path.join(rec_dir, 'txt')
            os.makedirs(txt_path, exist_ok=True)
        print(f'#### IMAGE: {bname}')

        # first image
        os.makedirs(os.path.join(args.output_dir, 'per-scale', 'scale0'), exist_ok=True)
        first_image = Image.open(image_name).convert('RGB')
        first_image.save(f'{rec_dir}/0.png')
        first_image.save(os.path.join(args.output_dir, 'per-scale', 'scale0', bname))

        w, h = first_image.size
        scale = 512 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        full_image = first_image.resize((new_w, new_h), Image.LANCZOS)
        # save 512x512 version of full image for VLM input
        full_image.save(f'{rec_dir}/0_full.png')

        # recursion
        for rec in range(args.rec_num):
            print(f'RECURSION: {rec}')
            os.makedirs(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}'), exist_ok=True)

            if args.rec_type == 'recursive_multiscale':
                full_path = f'{rec_dir}/0_full.png'
                curr_path = f'{rec_dir}/{rec}.png'
                next_path = f'{rec_dir}/{rec+1}.png'
                prev = Image.open(curr_path).convert('RGB')
                
                w, h = prev.size
                new_w, new_h = int(w * args.upscale), int(h * args.upscale)
                print('new_w, new_h:', new_w, new_h)
                prev = prev.resize((new_w, new_h), Image.LANCZOS)
                p = next(model.vae.parameters())
                x_full = transforms.ToTensor()(prev).unsqueeze(0).to(p.device, dtype=p.dtype) * 2 - 1
                
                full_latent, _ = model_test.create_full_latent(x_full, vlm_model, vlm_processor, full_path, next_path, args.prompt_type)
                out_img = model_test.decode_full_latent(full_latent).cpu()
                out_pil = transforms.ToPILImage()((out_img[0] * 0.5 + 0.5).clamp(0,1))
                out_pil.save(next_path)
                out_pil.save(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}', bname))
                for fp in glob.glob(os.path.join(rec_dir, '*patch*.png')):
                    if os.path.isfile(fp): os.remove(fp)

            else:
                raise ValueError(f"Unknown recursion_type: {args.rec_type}")

