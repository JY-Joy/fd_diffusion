import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch as th
import safetensors

from ema_pytorch import EMA
from decomp_diffusion.image_datasets import get_dataset
from decomp_diffusion.model_and_diffusion_util import *
from decomp_diffusion.diffusion.respace import SpacedDiffusion
from decomp_diffusion.gen_image import *
from accelerate.utils import set_seed


if __name__=='__main__':
    """
    Generate images. Several use cases:
        - Generate single image: provide --im_path
        - Generate multiple images from a dataset: provide --gen_images
        - Generate combinations from a dataset: provide --combine
        - Generate image components in separate files: provide --separate
    """
    defaults = model_defaults()
    defaults.update(unet_model_defaults())
    defaults.update(diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--ckpt_path2', default=None, help="for cross-dataset")

    parser.add_argument('--seed', default=3467, type=int)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--free', action='store_true')
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--im_path', default='clevr_im_10.png')
    parser.add_argument('--im_path2', default=None, help="for combination")

    parser.add_argument('--dataset', default='clevr')
    parser.add_argument('--dataset2', default=None) # for multi-modal combination

    parser.add_argument('--gen_images', default=0, type=int) # generate multiple images from dataset
    parser.add_argument('--sample_method', default='ddim')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--num_images', default=1, type=int) # if gen 1 img, how many samples
    parser.add_argument('--combine_method', default=None) # slice, add, or cross-dataset
    parser.add_argument('--indices', type=str, default=None) # comma-delimited list

    args = parser.parse_args()

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    set_seed(args.seed)

    ckpt_path = args.ckpt_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    guidance_scale = args.guidance_scale
    free = args.free
    dataset = args.dataset
    dataset2 = args.dataset2
    if dataset2 == None:
        dataset2 = dataset
    image_size = args.image_size
    sample_method = args.sample_method
    num_images = args.num_images
    indices = args.indices
    data_dir = args.data_dir

    model_desc = args.model_desc
    training_model_defaults = unet_model_defaults() if model_desc == 'unet_model' else model_defaults()
    model_kwargs = args_to_dict(args, training_model_defaults.keys())
    model = create_diffusion_model(**model_kwargs)
    if 'ema' in ckpt_path:
        model = EMA(model, beta=0.99, update_every=1).online_model

    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())
    gd = create_gaussian_diffusion(**diffusion_kwargs)

    model.eval()
    model.to(device)

    print(f'loading from {ckpt_path}')
    # Load distilled weights
    if ckpt_path.endswith('.safetensors'):
        state_dict = dict()
        with safetensors.safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = th.load(ckpt_path, map_location='cpu')

    model.load_state_dict(state_dict)
    print(f'saving to {save_dir}')
    gen_images = args.gen_images
    step_size = 100 if 'kitti' in dataset or 'falcor' in dataset else 1 # kitti, vkitti have very similar adjacent imgs
    if sample_method == 'ddim':
        # use respaced diffusion steps
        desired_timesteps = 50
        num_timesteps = diffusion_kwargs['steps']

        spacing = num_timesteps // desired_timesteps
        spaced_ts = list(range(0, num_timesteps + 1, spacing))
        betas = get_named_beta_schedule(diffusion_kwargs['noise_schedule'], num_timesteps)
        diffusion_kwargs['betas'] = betas
        del diffusion_kwargs['steps'], diffusion_kwargs['noise_schedule']
        gd = SpacedDiffusion(spaced_ts, rescale_timesteps=True, original_num_steps=num_timesteps, **diffusion_kwargs)

    if gen_images > 0: # generate multiple imgs
        num_digits = len(str(gen_images * step_size))
        data = get_dataset(dataset, base_dir=data_dir, num_images=gen_images * step_size, resolution=image_size).images
        data2 = get_dataset(dataset2, base_dir=data_dir, num_images=gen_images * step_size, resolution=image_size).images

    images = get_gen_images(
        model, gd, seed=args.seed,
        sample_method=sample_method, im_path=args.im_path, image_size=image_size,
        device=device, guidance_scale=guidance_scale, free=free,
        separate=args.separate,
    )

    if args.separate:
        masks = images[1]
        images = images[0]
    samples = th.cat(images, dim=0).cpu()
    if args.separate:
        masks = th.cat(masks, dim=0).cpu()
        mask_grid = make_grid(masks, nrow=samples.shape[0], padding=0)
        save_image(mask_grid, os.path.join(save_dir, f'{dataset}_mask_70k.png'))
    grid = make_grid(samples, nrow=samples.shape[0], padding=0)
    save_image(grid, os.path.join(save_dir, f'{dataset}_70k.png'))
