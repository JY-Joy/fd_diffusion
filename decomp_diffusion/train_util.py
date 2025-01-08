import torch as th
import argparse
import accelerate
import os
import numpy as np
import copy
from ema_pytorch import EMA
from tqdm.auto import tqdm

from .model_and_diffusion_util import *
from .gen_image import get_gen_images

def uniform_sample_timesteps(steps, batch_size):
    indices_np = np.random.choice(steps, size=(batch_size,))
    indices = th.from_numpy(indices_np).long() # .to(device)
    return indices

def update_ema(target_params, source_params, ema_rate=0.99):
    # update target in-place
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(ema_rate).add_(src, alpha=1 - ema_rate)

def params_to_state_dict(target_params, model):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = target_params[i]
    return state_dict

def run_loop(accelerator, model, gd, train_dataloader, optimizer, args, global_step=0, start_step=0, start_epoch=0, p_uncond=0.0, ddim_gd=None, latent_orthog=False, ema_rate=0.9999, dataset='clevr', downweight=False, image_size=64):

    # ddim sampling for generating samples per epoch block
    if ddim_gd == None:
        ddim_gd = create_ddim_diffusion(diffusion_defaults())

    # ema_params = copy.deepcopy(list(model.parameters()))
    ema_model = EMA(
        accelerator.unwrap_model(model),
        beta=ema_rate,
        update_every=1
    )
    ema_model.to(accelerator.device)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=start_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    use_CFG = p_uncond > 0
    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()
        for step, (batch, cond) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch = batch.to(accelerator.device)
                model_kwargs = {}

                if not use_CFG:
                    model_kwargs = dict(latent=None, x_start=batch)
                else:
                    # Condition dropout
                    b = batch.shape[0]
                    rand_values = th.rand(b, 1).to(accelerator.device)
                    keep_mask = rand_values >= p_uncond
                    null_emb = th.zeros(b, model.latent_dim_expand).to(accelerator.device)
                    latent_emb = model.encode_latent(batch)
                    emb = th.where(
                        keep_mask,
                        latent_emb,
                        null_emb
                    )
                    model_kwargs = dict(latent=emb)

                t = uniform_sample_timesteps(gd.num_timesteps, len(batch)).to(accelerator.device)

                loss = gd.training_losses(model, batch, t, model_kwargs=model_kwargs, latent_orthog=latent_orthog, downweight=downweight)
                loss = loss.mean() 
                accelerator.backward(loss)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.step()
                optimizer.zero_grad()

            # update ema params
            # update_ema(ema_params, list(model.parameters()), ema_rate=ema_rate)
            ema_model.update()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        th.save(ema_model.state_dict(), os.path.join(args.output_dir, f"checkpoint-{global_step}", f'ema_{ema_rate}_model_ckpt.pt'))
                        print(f"Saved state to {save_path}")
                if global_step % args.validation_steps == 0:
                    im_path=f"./val_imgs/{dataset}.jpg"
                    images = get_gen_images(
                        accelerator.unwrap_model(model), ddim_gd, image_size=image_size,
                        im_path=im_path, device=accelerator.device, free=use_CFG, sample_method='ddim',
                    )
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = th.cat(images, dim=0).cpu().numpy()
                            tracker.writer.add_images("validation", np_images, global_step, dataformats="NCHW")

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def create_ema(save_desc, epoch_block=10000, last_epoch=140000):
    save_dir = 'logs_' + save_desc
    defaults = model_defaults()

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--ema_rate', type=float, default=0.99)
    args = parser.parse_args()

    ema_rate = args.ema_rate
    model_kwargs = args_to_dict(args, model_defaults().keys())
    model = create_diffusion_model(**model_kwargs)
    model.eval()
    device = 'cuda'
    model.to(device)
    ema_params = 0

    for epoch in range(0, last_epoch + 1, epoch_block):
        ckpt_path = os.path.join(save_dir, f'model_{epoch}.pt')
        print(f'loading from {ckpt_path}')
        checkpoint = th.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        if epoch == 0:
            ema_params = copy.deepcopy(list(model.parameters()))
        update_ema(ema_params, list(model.parameters()), ema_rate=ema_rate)
        print(epoch)

        ema_state_dict = params_to_state_dict(ema_params, model)
        th.save(ema_state_dict, os.path.join(save_dir, f'ema_{ema_rate}_{epoch}.pt'))

if __name__=='__main__':
    save_desc = 'unet_model_celebahq_10000_xstart_emb_128'
    create_ema(save_desc)
