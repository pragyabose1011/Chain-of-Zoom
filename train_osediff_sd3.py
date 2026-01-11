import os
import gc
import lpips
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from copy import deepcopy
import itertools
import wandb

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from osediff_sd3 import OSEDiff_SD3_GEN, OSEDiff_SD3_REG, SD3Euler, add_mp_hook
from train_utils.realsr_dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from diffusers import StableDiffusion3Pipeline

from lora.lora_utils import save_lora_weight


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")

def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")

def parse_str_list(arg):
    return arg.split(',')

def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--output_dir", default='experience/osediff_sd3')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--learning_rate_vae", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--log_wandb", action="store_true",)
    
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff", help="The name of the wandb project to log to.")
    parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['YOUR TXT FILE PATH'], help='A comma-separated list of integers')
    parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1], help='A comma-separated list of integers')
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default='stabilityai/stable-diffusion-3-medium-diffusers', type=str)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)
    # ram path
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def count_parameters(model: torch.nn.Module) -> int:
    """
    Returns the total number of trainable parameters and total number of parameters in `model`.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints_reg"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints_vae"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualization/restoration"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualization/vsd"), exist_ok=True)

    # initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.tracker_project_name,
            config=vars(args),
            name=f"osediff_sd3"
        )

    #----------------------------------- MODEL -----------------------------------#
    model = SD3Euler()
    model.text_enc_1.to('cuda:0')
    model.text_enc_2.to('cuda:0')
    model.text_enc_3.to('cuda:0')
    model.transformer.to('cuda:2', dtype=torch.float32)
    model.vae.to('cuda:3', dtype=torch.float32)
    
    model.text_enc_1.requires_grad_(False)
    model.text_enc_2.requires_grad_(False)
    model.text_enc_3.requires_grad_(False)
    model.transformer.requires_grad_(False)
    model.vae.requires_grad_(False)

    # initialize OSEDiff_SD3_GEN and OSEDiff_SD3_REG with shared models
    model_gen = OSEDiff_SD3_GEN(args, model)
    model_reg = OSEDiff_SD3_REG(args, model)
    
    # for name, param in model_gen.transformer_gen.named_parameters():
    #     print(name, param.requires_grad)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.transformer_gen.enable_xformers_memory_efficient_attention()
            model_reg.transformer_reg.enable_xformers_memory_efficient_attention()
            model_reg.transformer_org.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.transformer_gen.enable_gradient_checkpointing()
        model_reg.transformer_reg.enable_gradient_checkpointing()
        model_reg.transformer_org.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    #----------------------------------- OPTIMIZER -----------------------------------#
    gen_lora_params = []
    for n, _p in model_gen.transformer_gen.named_parameters():
        if "lora" in n:
            gen_lora_params.append(_p)
    optimizer_gen = torch.optim.AdamW(gen_lora_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    
    reg_lora_params = []
    for n, _p in model_reg.transformer_reg.named_parameters():
        if "lora" in n:
            reg_lora_params.append(_p)
    optimizer_reg = torch.optim.AdamW(reg_lora_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    
    # initialize optimizer for VAE encoder
    encoder_params = []
    for n, _p in model.vae.named_parameters():
        if "lora" in n:
            encoder_params.append(_p)
    optimizer_encoder = torch.optim.AdamW(
        encoder_params,
        lr=args.learning_rate_vae,  # Use a separate learning rate if defined
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Count total parameters
    gen_num_params = sum(p.numel() for p in gen_lora_params)
    print(f"gen_lora_params total = {gen_num_params}")

    reg_num_params = sum(p.numel() for p in reg_lora_params)
    print(f"reg_lora_params total = {reg_num_params}")
    
    gen_trainable_params, gen_params = count_parameters(model_gen.transformer_gen)
    reg_trainable_params, reg_params = count_parameters(model_reg.transformer_reg)
    vae_trainable_params, vae_params = count_parameters(model.vae)

    print(f"Trainable params in model_gen: {gen_trainable_params:,} / {gen_params:,}")
    print(f"Trainable params in model_reg: {reg_trainable_params:,} / {reg_params:,}")
    print(f"Trainable params in model.vae: {vae_trainable_params:,} / {vae_params:,}")
    print(f"Total trainable params: {(gen_trainable_params + reg_trainable_params + vae_trainable_params):,} / {(gen_params + reg_params + vae_params):,}")

    #----------------------------------- DATASET -----------------------------------#
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    print(f'batch size: {args.train_batch_size}, gradient accumulation steps: {args.gradient_accumulation_steps}')
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    
    #----------------------------------- RAM MODEL -----------------------------------#
    # init ram model
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_ram = ram(pretrained=args.ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    model_ram.eval()
    model_ram.to("cuda:2")

    #----------------------------------- TRAIN -----------------------------------#
    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps")

    # Start the training loop
    global_step = 0
    accum_step = 0
    
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):

            x_src = batch["conditioning_pixel_values"]
            x_tgt = batch["output_pixel_values"]

            B, C, H, W = x_src.shape
            # Get text prompts from GT
            x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
            caption = inference(x_tgt_ram.to('cuda:2'), model_ram)
            batch["prompt"] = [f'{each_caption}' for each_caption in caption]
            
            # Forward pass
            x_tgt_pred, latents_pred, prompt_embeds, pooled_embeds = model_gen(x_src, batch=batch, args=args)
            
            #-------- Visualization --------#
            if global_step % 100 == 1:
                x_tgt_pil = transforms.ToPILImage()(x_tgt[0].cpu() * 0.5 + 0.5)
                x_src_pil = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                x_tgt_pred_clamp = torch.clamp(x_tgt_pred[0].cpu(), -1.0, 1.0)      # clamp
                x_tgt_pred_pil = transforms.ToPILImage()(x_tgt_pred_clamp * 0.5 + 0.5)
                # Concatenate images side by side
                w, h = x_tgt_pil.width, x_tgt_pil.height
                combined_image = Image.new('RGB', (w*3, h))
                combined_image.paste(x_tgt_pil, (0, 0))
                combined_image.paste(x_src_pil, (w, 0))
                combined_image.paste(x_tgt_pred_pil, (w*2, 0))
                combined_image_path = os.path.join(args.output_dir, f'visualization/restoration/{global_step}.png')
                combined_image.save(combined_image_path)
                
            # Log the image to wandb
            if args.log_wandb:
                if global_step % args.checkpointing_steps == 1:
                    wandb.log({"restoration_image": wandb.Image(combined_image_path)}, step=global_step)
            #-------- Visualization --------#

            # Reconstruction loss
            device = next(net_lpips.parameters()).device
            loss_l2 = F.mse_loss(x_tgt_pred.float().to(device), x_tgt.float().to(device), reduction="mean") * args.lambda_l2
            loss_lpips = net_lpips(x_tgt_pred.float().to(device), x_tgt.float().to(device)).mean() * args.lambda_lpips
            loss = loss_l2 + loss_lpips
            
            # KL loss
            loss_kl = model_reg.distribution_matching_loss(z0=latents_pred, prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, global_step=global_step, args=args) * args.lambda_vsd
            loss = loss + loss_kl.to(device)
            loss = loss / args.gradient_accumulation_steps      # Accumulate gradients
            loss.backward()

            """
            diff loss: train lora of reg_model
            """

            loss_d = model_reg.diff_loss(z0=latents_pred, prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, net_lpips=net_lpips, args=args)*args.lambda_vsd_lora
            loss_d = loss_d / args.gradient_accumulation_steps
            loss_d.backward()
            
            accum_step += 1

            if accum_step % args.gradient_accumulation_steps == 0:
                
                # Gradient Clipping / Optimizer Step for model_gen
                utils.clip_grad_norm_(model_gen.parameters(), max_norm=args.max_grad_norm)            
                optimizer_gen.step()
                optimizer_gen.zero_grad(set_to_none=args.set_grads_to_none)
                        
                utils.clip_grad_norm_(encoder_params, max_norm=args.max_grad_norm)
                optimizer_encoder.step()
                optimizer_encoder.zero_grad(set_to_none=args.set_grads_to_none)
                
                # Gradient Clipping / Optimizer Step for model_reg
                utils.clip_grad_norm_(model_reg.parameters(), max_norm=args.max_grad_norm)
                optimizer_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss_d": loss_d.detach().item(),
                    "loss_kl": loss_kl.detach().item(),
                    "loss_l2": loss_l2.detach().item(),
                    "loss_lpips": loss_lpips.detach().item(),
                    "global_step": global_step
                }
                progress_bar.set_postfix(**logs)
                if args.log_wandb:
                    wandb.log(logs, step=global_step)

            # checkpoint the model
            if global_step % args.checkpointing_steps == 1:
                
                # 1. Save transformer_gen
                out_gen = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                # remove forward hook temporarily
                for hook in model_gen.hooks:
                    hook.remove()
                # save
                save_lora_weight(model_gen.transformer_gen, out_gen)
                # add forward hook again
                model_gen.transformer_gen, model_gen.hooks = add_mp_hook(model_gen.transformer_gen)
                
                # 2. Save transformer_reg
                out_reg = os.path.join(args.output_dir, "checkpoints_reg", f"model_{global_step}.pkl")
                # remove forward hook temporarily
                for hook in model_reg.hooks:
                    hook.remove()
                # save
                save_lora_weight(model_reg.transformer_reg, out_reg)
                # add forward hook again
                model_reg.transformer_reg, model_reg.hooks = add_mp_hook(model_reg.transformer_reg)
                
                # 3. Save encoder weights
                
                # Create a copy of the encoder's state_dict in float16
                encoder_state_dict_fp16 = {k: v.half() for k, v in model.vae.encoder.state_dict().items()}

                # Save the float16 state_dict
                out_encoder = os.path.join(args.output_dir, "checkpoints_vae", f"vae_encoder_{global_step}.pt")
                torch.save(encoder_state_dict_fp16, out_encoder)
    if args.log_wandb:
        wandb.finish()                


if __name__ == "__main__":
    args = parse_args()
    main(args)
