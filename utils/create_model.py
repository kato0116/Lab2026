import torch


def create_model(args):
    if args.model_name =="ssdim":
        from models.ssdit_mamba import SSDiM, Model_Size
        # from models.ssdit import SSDiT, Model_Size
        model_config = Model_Size[args.model_size]
        model = SSDiM(
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
        
        # モデルのパラメータ数計測
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model {args.model_name} has {num_params/1e6:.2f} million parameters.")
        # def forward(self, x, t, conditioned_image):

        
        
        # 学習時のGPUメモリ使用量の計測
        from torch.cuda import memory_allocated, max_memory_allocated
        model.to('cuda')
        dummy_x = torch.randn(1, args.latent_dim, args.latent_size, args.latent_size).to('cuda')
        dummy_t = torch.tensor([0]).to('cuda')  # ダミーの
        dummy_cond = torch.randn(1, args.img_channels, args.img_size, args.img_size).to('cuda')
        
        # 学習時
        out = model(dummy_x, dummy_t, dummy_cond)
        train_memory = max_memory_allocated() / 1e9  # GB単位に変換
        print(f"Model {args.model_name} uses {train_memory:.2f} GB of GPU memory during training.")
        # 推論時のメモリ使用量の計測
        with torch.no_grad():
            out = model(dummy_x, dummy_t, dummy_cond)
        inference_memory = max_memory_allocated() / 1e9  # GB単位
        print(f"Model {args.model_name} uses {inference_memory:.2f} GB of GPU memory during inference.")
        # メモリ解放
        del dummy_x, dummy_t, dummy_cond, out
        torch.cuda.empty_cache()
        
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
