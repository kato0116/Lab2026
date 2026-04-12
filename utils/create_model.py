import torch


def create_model(args):
    if args.model_name =="ssdim":
        from models.ssdit_mamba import SSDiT, Model_Size
        # from models.ssdit import SSDiT, Model_Size
        model_config = Model_Size[args.model_size]
        model = SSDiT(
            input_size       =args.latent_size,         # 32
            patch_size       =args.patch_size,          # 2
            in_channels      =args.latent_dim,          # 4
            cond_in_channels =args.img_channels,        # 3
            hidden_size =model_config["hidden_size"],   # 384
            depth       =model_config["depth"],         # 12
            num_heads   =model_config["num_heads"],     # 4
            skip_flag   =args.skip_flag,                # False
            unet_hidden_size =args.unet_hidden_size,    # 64
            cross_attn_flag  =args.cross_attn_flag,     # False
            shared_step      =args.shared_step,         # False
            )
    elif args.model_name =="ssdit":
        from models.ssdit import SSDiT, Model_Size
        # from models.ssdit import SSDiT, Model_Size
        model_config = Model_Size[args.model_size]
        model = SSDiT(
            input_size       =args.latent_size,         # 32
            patch_size       =args.patch_size,          # 2
            in_channels      =args.latent_dim,          # 4
            cond_in_channels =args.img_channels,        # 3
            hidden_size =model_config["hidden_size"],   # 384
            depth       =model_config["depth"],         # 12
            num_heads   =model_config["num_heads"],     # 4
            skip_flag   =args.skip_flag,                # False
            unet_hidden_size =args.unet_hidden_size,    # 64
            cross_attn_flag  =args.cross_attn_flag,     # False
            shared_step      =args.shared_step,         # False
            )
    elif args.model_name == "medsegdiff":
        from medsegdiff.script_util import (
            model_defaults,
            create_model,
        )
        model = create_model(**model_defaults())
    elif args.model_name == "lsegdiff":
        from models.lsegdiff_original import LSegUNet
        model = LSegUNet(
            in_channels=args.latent_dim,
            cond_in_channels=args.img_channels,
            unet_hidden_size=128,
            time_embed_dim=256,
        )
    else:
        raise NotImplementedError(f"Model {args.model_name} is not implemented.")
    return model
