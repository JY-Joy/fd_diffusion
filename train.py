"""
Train a diffusion model on images.
"""
import os

import argparse
import logging
import json
import safetensors

import torch as th

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from decomp_diffusion.image_datasets import load_data, get_dataset
from decomp_diffusion.model import *
from decomp_diffusion.train_util import run_loop
from decomp_diffusion.diffusion.gaussian_diffusion import *
from decomp_diffusion.model_and_diffusion_util import *

import PIL
from packaging import version

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("10.0.0"):
    PIL.Image.ANTIALIAS=PIL.Image.LANCZOS

logger = get_logger(__name__)


def main():
    args = create_argparser().parse_args()

    # Trial description
    model_desc = args.model_desc
    num_images = int(args.num_images) if (args.num_images is not None) else None
    predict_mean = args.predict_xstart
    dataset = args.dataset
    downweight = args.downweight
    image_size = args.image_size

    predict_type = 'xstart' if predict_mean else 'eps'
    trial_desc = f'{dataset}_{predict_type}_emb_{args.emb_dim}_comp_{args.num_components}'
    if args.p_uncond > 0:
        trial_desc += f'_CFG{args.p_uncond}'
    if len(args.extra_desc) > 0:
        trial_desc += '_' + args.extra_desc
    args.output_dir = os.path.join(args.output_dir, trial_desc)

    logging_dir = os.path.join(args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)  

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # create model
    logger.info("creating model and diffusion...")
    training_model_defaults = unet_model_defaults()
    model_kwargs = args_to_dict(args, training_model_defaults.keys())
    model = create_diffusion_model(**model_kwargs)

    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())
    gd = create_gaussian_diffusion(**diffusion_kwargs)

    # load pre-trained weights
    if args.ckpt_path is not None:
        print(f'loading from {args.ckpt_path}')
        # Load distilled weights
        if args.ckpt_path.endswith('.safetensors'):
            state_dict = dict()
            with safetensors.safe_open(args.ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = th.load(args.ckpt_path, map_location='cpu')

        model.load_state_dict(state_dict)

    relevant_keys = list(model_defaults().keys()) +  list(training_defaults().keys())
    key_args = args_to_dict(args, relevant_keys)
    json.dump(key_args, open(os.path.join(args.output_dir, 'arguments.json'), "w"), sort_keys=True, indent=4)

    # optimizer
    optimizer = th.optim.Adam(
        model.parameters(),  # only optimize the embeddings
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Data
    logger.info("creating data loader...")
    train_dataset = get_dataset(base_dir=args.data_dir, dataset_type=dataset, num_images=num_images, resolution=image_size)
    train_dataloader = th.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = th.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = th.float16

    # Move vae and unet to device and cast to weight_dtype
    model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("decomp_diff", config=key_args)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    run_loop(
        accelerator, args,
        model, gd, train_dataloader, optimizer,
        global_step=global_step, start_step=initial_global_step, start_epoch=first_epoch,
        dataset=dataset, image_size=image_size,
        p_uncond=args.p_uncond, regu_weight=args.regu_weight,
        latent_orthog=args.latent_orthog, downweight=downweight,
    )

def parse_epoch(ckpt_path):
    """ckpt path must be {save_dir}/model_{epoch}.pt or {save_dir}/ema_{ema_rate}_{epoch}.pt"""
    assert ckpt_path[-3:] == '.pt'
    end_idx = -3 # exclusive
    start_idx = end_idx
    while ckpt_path[start_idx] != '_':
        start_idx -= 1
    start_idx += 1 # start of num
    epoch = int(ckpt_path[start_idx : end_idx])
    return epoch

def training_defaults():
    # directory configs
    return dict(
        p_uncond=0.0,
        regu_weight=0.0,
    )

def create_argparser():
    # Regular hyperparameters
    base_args = dict(
        data_dir="",
        dataset="clevr",
        dataloader_num_workers=8,
        output_dir="output",
        report_to="wandb",
        logging_dir="logs",
        schedule_sampler="uniform",
        seed=3467,
        lr=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.0,
        adam_epsilon=1e-08,
        gradient_accumulation_steps=1,
        num_train_epochs=110,
        max_train_steps=40000,
        checkpointing_steps=10000,
        validation_steps=500,
        lr_anneal_steps=0,
        train_batch_size=1,
        ema_rate="0.9999",  # comma-separated list of EMA values
        resume_from_checkpoint="latest",
        ckpt_path=None,
        num_images=None,
        downweight=False,
        mixed_precision='no',
        latent_orthog=False,
        extra_desc='',
    )
    base_args.update(training_defaults())
    base_args.update(diffusion_defaults())
    base_args.update(unet_model_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, base_args)
    return parser


if __name__ == "__main__":
    main()
