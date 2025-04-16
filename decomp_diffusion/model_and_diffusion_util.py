from .diffusion.gaussian_diffusion import get_named_beta_schedule, GaussianDiffusion
from .model.unet import UNetModel
from .diffusion.respace import SpacedDiffusion

def create_unet_model(
        in_channels=3,
        image_size=64,
        num_channels=64,
        enc_channels=64,
        encoder_channels=128,
        num_res_blocks=2,
        num_components=4, 
        channel_mult="",
        encoder_channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        steps=1000,
        use_scale_shift_norm=False,
        resblock_updown=True,
        model_desc='unet_model',
        emb_dim=256,
        time_embed_dim=128
    ):
        # everything else False

    if channel_mult == "":
        if image_size == 64:
            channel_mult = (1, 2, 3) # (1, 2, 3, 4)
            encoder_channel_mult = (2, 3)
        elif image_size == 128:
            channel_mult = (1, 2, 3, 4)
            encoder_channel_mult = (2, 3, 4)
        elif image_size < 64: # eg 35
            channel_mult = (1, 2)
            encoder_channel_mult = (2, 3)
    elif len(channel_mult) > 0: # passed in comma-delimited series of numbers
        channel_mult = channel_mult.split(',')
        channel_mult = [int(n) for n in channel_mult]
        channel_mult = tuple(channel_mult)
        encoder_channel_mult = encoder_channel_mult.split(',')
        encoder_channel_mult = [int(n) for n in encoder_channel_mult]
        encoder_channel_mult = tuple(encoder_channel_mult)
        
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    unet = UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        enc_channels=enc_channels,
        out_channels=in_channels,
        num_res_blocks=num_res_blocks,
        num_components=num_components,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        encoder_channel_mult=encoder_channel_mult,
        num_timesteps=steps,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        encoder_channels=encoder_channels,
        image_size=image_size,
        emb_dim=emb_dim,
        time_embed_dim=time_embed_dim,
    )
    return unet

def unet_model_defaults():
    return dict(
        in_channels=3,
        image_size=64,
        num_channels=64,
        enc_channels=64,
        encoder_channels=128,
        num_res_blocks=2,
        num_components=4,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        steps=1000,
        use_scale_shift_norm=False,
        resblock_updown=True,
        model_desc='unet_model',
        emb_dim=256,
        time_embed_dim=64,
    )

def create_diffusion_model(model_desc='unet_model', **model_kwargs):
    if model_desc == 'unet_model':
        model = create_unet_model(**model_kwargs)
    else:
        raise NotImplementedError(f"Model {model_desc} not implemented")
    return model


def create_gaussian_diffusion(steps=1000, noise_schedule="squaredcos_cap_v2", predict_xstart=True):
    betas = get_named_beta_schedule(noise_schedule, steps)
    gd = GaussianDiffusion(betas, predict_xstart=predict_xstart)
    return gd

def create_ddim_diffusion(diffusion_kwargs, desired_timesteps=50):
    # use respaced diffusion steps
    num_timesteps = diffusion_kwargs['steps']

    spacing = num_timesteps // desired_timesteps
    spaced_ts = list(range(0, num_timesteps + 1, spacing))
    betas = get_named_beta_schedule(diffusion_kwargs['noise_schedule'], num_timesteps)
    diffusion_kwargs['betas'] = betas
    del diffusion_kwargs['steps'], diffusion_kwargs['noise_schedule']
    ddim_gd = SpacedDiffusion(spaced_ts, rescale_timesteps=True, original_num_steps=num_timesteps, **diffusion_kwargs)
    return ddim_gd

def model_defaults():
    return dict(
        in_channels=3,                  # overwrote by unet_model_defaults
        filter_dim=16,                  # deprecated
        emb_dim=256,                    # overwrote by unet_model_defaults
        num_components=4,               # overwrote by unet_model_defaults
        model_desc='Segment_diffusion', # overwrote by unet_model_defaults
        image_size=64 # added           # overwrote by unet_model_defaults
    )

def diffusion_defaults():
    return dict(
        steps=1000,
        noise_schedule="squaredcos_cap_v2",
        predict_xstart=True
    )


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}
