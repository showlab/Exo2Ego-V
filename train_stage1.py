import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from get_dataset import EgoExo4Ddataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_image import Exo2EgoImagePipeline
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything, _resize_with_antialiasing
from torch.utils.data import RandomSampler
from src.render import NeRFRenderer
from args import parse_args
from train_ego import PixelNeRFTrainer
from src.model import make_model, loss
import pdb
from transformers import CLIPImageProcessor

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        clip_image_embeds_exo,
        rel_poses,
        uncond_fwd: bool = False,
    ):

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds_exo,
                poses=rel_poses.reshape(rel_poses.shape[0], -1),
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=None,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    pixelnerf_features = None,
    pixel_values = None,
    poses=None,
    condition_values = None, 
    nerf_outputs = None,
):
    logger.info("Running validation... ")
    import copy

    ori_net = copy.deepcopy(accelerator.unwrap_model(net))
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet

    generator = torch.manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    image_enc = image_enc.to(dtype=torch.float32)

    pipe = Exo2EgoImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)


    pil_images = []
    eval_name = "eval"
    image = pipe(
        condition_values,
        width,
        height,
        20,
        1.5,
        poses=poses,
        generator=generator,
        pixelnerf_features = pixelnerf_features,
        nerf_outputs = nerf_outputs,
    ).images
    image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
    res_image_pil = Image.fromarray((image * 255).astype(np.uint8))

    w, h = res_image_pil.size
    canvas = Image.new("RGB", (w * 2, h), "white")
    r1 = (pixel_values+1).squeeze().permute(1,2,0).cpu().detach().numpy()  
    r1 = Image.fromarray((r1 * 127.5).astype('uint8')) 
    canvas.paste(r1, (0, 0))
    canvas.paste(res_image_pil, (w , 0))

    pil_images.append({"name": f"{eval_name}", "img": canvas})
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipe
    torch.cuda.empty_cache()

    return pil_images


def main(cfg,args,conf):

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{args.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    if args.use_pixelnerf:
        pixelNeRF = make_model(conf["model"]).to(device=accelerator.device)
        if args.nerf_path is not None:
            pixelNeRF.load_state_dict(torch.load(args.nerf_path,map_location = "cpu"),strict=True)
        pixelNeRF.stop_encoder_grad =  args.freeze_enc
        renderer = NeRFRenderer.from_conf(conf["renderer"]).to(
            device=accelerator.device
        )
        render_par = renderer.bind_parallel(pixelNeRF, args.gpu_id)

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        [
                    {'params': trainable_params, 'lr': args.learning_rate},
                    {'params': pixelNeRF.parameters(), 'lr': args.nerf_learning_rate}
            ],
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = EgoExo4Ddataset(args = args, sample_frames=args.num_frames)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        if dirs != []:
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
            accelerator.load_state(os.path.join(resume_dir, path))
            accelerator.print(f"Resuming from checkpoint {path}")
            global_step = int(path.split("-")[1])

            pixel_nerf_path =  os.path.join(resume_dir, path, "pixelnerf", "model.pth")
            pixelNeRF.load_state_dict(torch.load(pixel_nerf_path).state_dict())        

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    if args.use_pixelnerf: 
        trainer = PixelNeRFTrainer(pixelNeRF, conf, renderer=renderer, render_par=render_par, args = args, device=accelerator.device)
        all_validation_values = train_dataset.get_validation(args,accelerator)

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            all_validation_values = train_dataset.get_validation(args,accelerator)
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch['pixel_values'][:,0,...].to(weight_dtype)
            if step%args.only_nerf_training==0:
                args.render_whole = False
            else:
                args.render_whole = True

            if args.use_pixelnerf:  
                if args.render_whole == True:
                    features = []
                    nerf_outputs = []
                    for i in range(args.num_frames):
                        ego_pose = batch['ego_poses'][0][i][None]
                        exo_poses = batch['exo_poses'][0]
                        all_poses = torch.cat((exo_poses,ego_pose),dim=0)[None]
                        all_images = [images[0][i] for images in batch['condition_values']]
                        all_images = torch.stack(all_images)[None]
                        ego_img = batch['ego_images'][:,i,...]
                        data = {
                            "focal": batch['focal'],
                            'images': all_images,
                            'poses': all_poses,
                            'c': batch['c'],
                            'ego_img': ego_img,
                            'scene': batch['scene'],
                            'ego_c': batch['c'],
                            'ego_focal': batch['focal']
                        }
                        render_dict,features_final,rgb_loss,nerf_output = trainer.calc_losses(data,step)
                        features.append(features_final)
                        nerf_outputs.append(nerf_output[0])      
                        
                    features_final = torch.stack(features).squeeze()
                    nerf_outputs = torch.stack(nerf_outputs)            

                else:
                    ego_pose = batch['ego_poses'][0][0][None]
                    exo_poses = batch['exo_poses'][0]
                    all_poses = torch.cat((exo_poses,ego_pose),dim=0)[None]
                    all_images = [images[0][0] for images in batch['condition_values']]
                    all_images = torch.stack(all_images)[None]
                    ego_img = batch['ego_images'][:,0,...]
                    data = {
                        "focal": batch['focal'],
                        'images': all_images,
                        'poses': all_poses,
                        'c': batch['c'],
                        'ego_img': ego_img,
                        'scene': batch['scene'],
                        'ego_c': batch['c'],
                        'ego_focal': batch['focal']
                    }

                    render_dict,features_final,rgb_loss,_ = trainer.calc_losses(data,step)
                
                if args.render_whole == False:
                    loss = rgb_loss 
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:

                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                        latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    if cfg.noise_offset > 0.0:
                        noise += cfg.noise_offset * torch.randn(
                            (noise.shape[0], noise.shape[1], 1, 1, 1),
                            device=noise.device,
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each video
                    timesteps = torch.randint(
                        0,
                        train_noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    uncond_fwd = random.random() < cfg.uncond_ratio
                    clip_image_list = []
                    ref_image_list = []

                    with torch.no_grad():

                        cropped_height, cropped_width = args.exo_cropped_height, args.exo_cropped_width
                        start_height = (args.exo_height-cropped_height)//2
                        start_width = (args.exo_width-cropped_width)//2

                        ref_img = torch.cat(batch['condition_values'])[:, 0, ...][:, :, start_height:args.exo_height-start_height, start_width:args.exo_width-start_width].to(weight_dtype)
                        ref_image_latents = vae.encode(
                            ref_img
                        ).latent_dist.sample()  # (bs, d, 64, 64)

                        ref_image_latents = ref_image_latents * 0.18215
                        nerf_outputs_interp = torch.clamp(_resize_with_antialiasing(nerf_outputs, (224, 224), interpolation = "bilinear"), min=0, max=1)
                        clip_image_processor = CLIPImageProcessor()
                        if uncond_fwd:
                            clip_img = torch.zeros_like(nerf_outputs_interp)
                        else:
                            clip_img = nerf_outputs_interp
                        clip_img = clip_image_processor(clip_img, return_tensors="pt",do_rescale=False).pixel_values

                        clip_image_embeds = image_enc(clip_img.to("cuda", dtype=weight_dtype)).image_embeds
                        image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                        if uncond_fwd:
                            clip_img_exo = torch.zeros_like(ref_img)
                        else:
                            clip_img_exo = (ref_img + 1) / 2.0
                        clip_img_exo = clip_image_processor(clip_img_exo, return_tensors="pt",do_rescale=False).pixel_values

                        clip_image_embeds_exo = image_enc(clip_img_exo.to("cuda", dtype=weight_dtype)).image_embeds
                        image_prompt_embeds_exo = clip_image_embeds_exo.unsqueeze(1)  # (bs, 1, d)

                    # add noise
                    noisy_latents = train_noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    if args.use_pixelnerf:
                        noisy_latents = torch.cat((noisy_latents,features_final[None,:,None,:,:]),dim=1)

                    # Get the target for loss depending on the prediction type
                    if train_noise_scheduler.prediction_type == "epsilon":
                        target = noise
                    elif train_noise_scheduler.prediction_type == "v_prediction":
                        target = train_noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                        )

                    model_pred = net(
                        noisy_latents,
                        timesteps,
                        ref_image_latents,
                        image_prompt_embeds,
                        image_prompt_embeds_exo,
                        batch["rel_poses"][0],                        
                        uncond_fwd
                    )

                    if cfg.snr_gamma == 0:
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        snr = compute_snr(train_noise_scheduler, timesteps)
                        if train_noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                            torch.stack(
                                [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                            ).min(dim=1)[0]
                            / snr
                        )
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                    train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_params,
                            cfg.solver.max_grad_norm,
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 5)
                        accelerator.save_state(save_path)
                        dir_path = os.path.join(save_path, "pixelnerf")
                        os.makedirs(dir_path, exist_ok=True)
                        torch.save(pixelNeRF, os.path.join(dir_path,"model.pth"))
    
                        
                if global_step % args.test_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        with torch.no_grad():
                            args.render_whole = True
                            pixelNeRF.eval()
                            validation_values_ego = all_validation_values['all_validation_values']['ego'][0].to(accelerator.device)
                            validation_values_exo = torch.stack(all_validation_values['all_validation_values']['exo'][0]).to(accelerator.device)
                            features = []
                            nerf_outputs = []

                            for i in range(args.num_frames):
                                all_images = [images[0][i] for images in all_validation_values['all_validation_values']['exo'][0]]
                                all_images = torch.stack(all_images)[None].to(accelerator.device)
                                all_poses = all_validation_values['exo_poses'].to(accelerator.device)
                                ego_pose = all_validation_values['ego_poses'][0,i:i+1,].to(accelerator.device)
                                all_poses = torch.cat((all_poses,ego_pose),dim=0).to(accelerator.device)
                                data_test = {
                                    "focal": all_validation_values['focal'][None].to(accelerator.device),
                                    'images': all_images,
                                    'poses': all_poses[None],
                                    'c': all_validation_values['c'][None].to(accelerator.device),
                                    'ego_img': all_validation_values['ego_images'][0][None].to(accelerator.device),
                                    'scene': [all_validation_values['scene']],
                                    'ego_c': all_validation_values['c'][None].to(accelerator.device),
                                    'ego_focal': all_validation_values['focal'][None].to(accelerator.device)
                                }
                                render_dict,pixelnerf_feature,rgb_loss,nerf_output = trainer.calc_losses(data_test,step)
                                features.append(pixelnerf_feature)
                                nerf_outputs.append(nerf_output[0])
                            pixelnerf_feature = torch.stack(features).squeeze()

                            nerf_outputs = torch.stack(nerf_outputs)
                            vis, vis_vals = trainer.vis_step(
                                data_test, global_step = 0
                            )
                            output_dir = os.path.join(args.output_dir,"nerf_test")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if vis is not None:
                                vis_u8 = (vis * 255).astype(np.uint8)
                                import imageio
                                imageio.imwrite(
                                    os.path.join(
                                        output_dir,
                                        "{:04}_vis.png".format(global_step),
                                    ),
                                    vis_u8,
                                )
                            val_img_idx = 0

                        nerf_outputs_interp = torch.clamp(_resize_with_antialiasing(nerf_outputs, (224, 224), interpolation = "bilinear"), min=0, max=1)

                        cropped_height, cropped_width = args.exo_cropped_height,args.exo_cropped_width
                        start_height = (args.exo_height-cropped_height)//2
                        start_width = (args.exo_width-cropped_width)//2

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=256,
                            height=256,
                            pixelnerf_features= pixelnerf_feature,
                            pixel_values = validation_values_ego[0],
                            poses = all_validation_values['rel_poses'].reshape(all_validation_values['rel_poses'].shape[0], -1).to(accelerator.device),
                            condition_values = validation_values_exo[:, 0, 0, ...][:, :, start_height:args.exo_height-start_height, start_width:args.exo_width-start_width],
                            nerf_outputs = nerf_outputs_interp,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            img = sample_dict["img"]
                            temp_dir = os.path.join(args.output_dir,"test_imgs")
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)
                            out_file = Path(
                                f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                            )
                            img.save(out_file)
                            mlflow.log_artifact(out_file)
                

                if global_step % args.validation_steps == 0:

                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        with torch.no_grad():
                            args.render_whole = True
                            pixelNeRF.eval()
                            features = []
                            nerf_outputs = []
                            
                            for i in range(args.num_frames):
                                ego_pose = batch['ego_poses'][0][i][None]
                                exo_poses = batch['exo_poses'][0]
                                all_poses = torch.cat((exo_poses,ego_pose),dim=0)[None]
                                all_images = [images[0][i] for images in batch['condition_values']]
                                all_images = torch.stack(all_images)[None]
                                ego_img = batch['ego_images'][:,i,...]
                                data = {
                                    "focal": batch['focal'],
                                    'images': all_images,
                                    'poses': all_poses,
                                    'c': batch['c'],
                                    'ego_img': ego_img,
                                    'scene': batch['scene'],
                                    'ego_c': batch['c'],
                                    'ego_focal': batch['focal']
                                }
                                render_dict,features_final,rgb_loss,nerf_output = trainer.calc_losses(data,step)

                                features.append(features_final)
                                nerf_outputs.append(nerf_output[0])

                            pixelnerf_feature = torch.stack(features).squeeze()
                            nerf_outputs = torch.stack(nerf_outputs)

                            vis, vis_vals = trainer.vis_step(
                                data,global_step = 0
                            )
                            output_dir = os.path.join(args.output_dir,"nerf")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if vis is not None:
                                vis_u8 = (vis * 255).astype(np.uint8)
                                import imageio
                                imageio.imwrite(
                                    os.path.join(
                                        output_dir,
                                        "{:04}_vis.png".format(global_step),
                                    ),
                                    vis_u8,
                                )
                            val_img_idx = 0
                        
                        nerf_outputs_interp = torch.clamp(_resize_with_antialiasing(nerf_outputs, (224, 224), interpolation = "bilinear"), min=0, max=1)

                        cropped_height, cropped_width = args.exo_cropped_height,args.exo_cropped_width
                        start_height = (args.exo_height-cropped_height)//2
                        start_width = (args.exo_width-cropped_width)//2

                        condition_values_ = torch.cat(batch['condition_values'])[:, 0, ...][:, :, start_height:args.exo_height-start_height, start_width:args.exo_width-start_width]
                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=256,
                            height=256,
                            pixelnerf_features= pixelnerf_feature,
                            pixel_values = batch['pixel_values'][:,0,...],
                            poses = batch["rel_poses"][0].reshape(batch["rel_poses"][0].shape[0], -1),
                            condition_values = condition_values_,
                            nerf_outputs = nerf_outputs_interp,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            img = sample_dict["img"]
                            temp_dir = os.path.join(args.output_dir,"validation_imgs")
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)
                            out_file = Path(
                                f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                            )
                            img.save(out_file)
                            mlflow.log_artifact(out_file)

                

                global_step += 1

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=5,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=5,
            )
            dir_path = os.path.join(save_dir, "pixelnerf")
            os.makedirs(dir_path, exist_ok=True)
            torch.save(pixelNeRF, os.path.join(dir_path, "model.pth"))

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    args , conf= parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config,args,conf)
