from utils.diffusion_script_util import select_type
import os
import argparse
import torch
from utils import create_folder, get_dataset, create_model, set_seed, set_name, Trainer
from torch.optim import AdamW
import numpy as np
import random


def main(args):
    set_seed(args.seed)        # シードを固定
    model = create_model(args) # モデルの作成
    args, dir_path = set_name(args)
    create_folder(dir_path)    # フォルダの作成

    
    # from diffusers import DDIMScheduler, DDIMInverseScheduler
    # scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.0001,beta_end=0.02,beta_schedule='linear')
    # inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',subfolder='scheduler')
    
    
    if args.diffuser_type == None: # 拡散モデルではない場合
        diffuser = None
    else:
        diffuser  = select_type(args.diffuser_type, args) # 拡散モデルの選択
    optimizer = AdamW(model.parameters(), lr=args.lr)     # 最適化手法
    
    # データセットの取得
    train_set, val_set, test_set = get_dataset(args)
    args.train_size = len(train_set)
    args.val_size   = len(val_set)
    args.test_size  = len(test_set)
    
    # 学習率スケジューラの設定
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-5)
    scheduler = None
    
    para            = sum([np.prod(list(p.size())) for p in model.parameters()])
    trainable_param = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model {args.model_name} has {para} parameters, trainable parameters: {trainable_param}")
    print(f"Model {args.model_name} has {para/1e6:.2f}M parameters, trainable parameters: {trainable_param/1e6:.2f}M")
    
    print("----- Training Configuration -----")
    print(f"Model: {args.model_name}")
    print(f"Train size: {args.train_size}, Val size: {args.val_size}, Test size: {args.test_size}")
    print(f"Fold: {args.fold}, Diffuser type: {args.diffuser_type}, Space: {args.space}")
    print(f"Learning Rate: {args.lr}, Epochs: {args.epochs}, Global Batch Size: {args.global_batch_size}, Microbatch: {args.microbatch}")
    print("----------------------------------")
    
    trainer = Trainer(
        model           =model,
        diffuser        =diffuser,
        optimizer       =optimizer,
        train_set       =train_set,
        val_set         =val_set,
        test_set        =test_set,
        args            =args,
        dir_path        =dir_path,
        scheduler       =scheduler,
        space           =args.space,
        parameter_num   =para,
        trainable_param =trainable_param,
        multi_gpu       =args.multi_gpu,
    )
    trainer.train_loop(args)
    trainer.test_loop(model, args)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default="/root/volume")
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--space', type=str, default='latent') # latent or pixel
    parser.add_argument('--diffuser_type', type=str, default='rectified_flow') # diffusion or rectified_flow or flow_matching

    # 学習パラメータ
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--global_batch_size', type=int, default=16)
    parser.add_argument('--microbatch', type=int, default=16)
    
    # データセットの設定
    parser.add_argument('--data_path', type=str, default="/root/volume/dataset/ISIC2018") # REFUGE-Multirater or REFUGE2_ver_crop_check or ISIC2018
    parser.add_argument('--dataset', type=str, default='ISIC2018') # ISIC2018 or REFUGE1_Crop or REFUGE-Disc or REFUGE2 or REFUGE2_Crop
    parser.add_argument('--center', type=bool, default=False)      # 中心を切り取るかどうか
    parser.add_argument('--crop', type=bool, default=False)
    parser.add_argument('--split_ODOC', type=str, default="")      # REFUGE2の分割 optic_cup or optic_disc
    parser.add_argument('--fold', type=str, default='0')           # REFUGEはfoldを指定なし
    parser.add_argument('--ch3_to_ch1', type=bool, default=False)  # ch3のvaeを使用してch1に変換するかどうか

    # モデルパスの選択
    parser.add_argument('--use_best_model', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--checkpoint',type=bool, default=False)
    parser.add_argument('--checkpoint_epoch', type=int, default=0)
    parser.add_argument('--save_n_model', type=int, default=500)
    parser.add_argument('--val_step_num', type=int, default=50)  # 何epochごとにvalidationを行うか
    parser.add_argument('--latent_sample', type=str, default='sample') # mean or sample
    
    # wandbの設定
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='SSDiM')
    # parser.add_argument('--test_project_name', type=str, default='SCI特集号_test')
    parser.add_argument('--model_name', type=str, default='ssdim')
    parser.add_argument('--notes', type=str, default='') # 実験の詳細
    parser.add_argument('--wandb_num_images', type=int, default=8)
    
    # SSDiTの設定
    """
    S:  {"hidden_size": 384, "depth": 12, "num_heads": 6}
    B:  {"hidden_size": 768, "depth": 12, "num_heads": 12}
    L:  {"hidden_size": 1024, "depth": 24, "num_heads": 16}
    XL: {"hidden_size": 1152, "depth": 28, "num_heads": 16}
    """
    parser.add_argument('--cond', type=str, default='ssdim')
    parser.add_argument('--patch_size', type=int, default=2)     # 2, 4, 8
    parser.add_argument('--model_size', type=str, default='S', choices=['XS', 'S', 'B', 'L', 'XL'])
    parser.add_argument('--unet_hidden_size', type=int, default=64)     # U-Netのデフォルトの隠れ層のサイズ
    parser.add_argument('--skip_flag', type=bool, default=False)         # skip
    parser.add_argument('--cross_attn_flag', type=bool, default=False)  # cross attention
    parser.add_argument('--shared_step', type=bool, default=True)       # shared step
    parser.add_argument('--image_diff', type=bool, default=False) # 画像の差分を取るかどうか
    
    # Mambaの設定
    parser.add_argument('--scan_type', type=str, default="zigzagN8") # zigzagN8 or raster
    parser.add_argument('--use_mamba2', type=bool, default=False)    # Mamba2を使用するかどうか
    parser.add_argument('--expand', type=int, default=1)             # Mambaのexpand
    parser.add_argument('--d_state', type=int, default=16)           # Mambaのd_state

    # 拡散モデルの共通パラメータ
    parser.add_argument('--val_ensemble', type=int, default=5)     # アンサンブル数
    parser.add_argument('--test_ensemble', type=int, default=15)   # テスト時のアンサンブル数
    parser.add_argument('--staple_flag', type=bool, default=False) # stapleを行うかどうか
    
    # 拡散モデルの設定
    parser.add_argument('--ddim', type=str, default='ddim50') # ddim100 or ddim50
    parser.add_argument('--predict_xstart', type=bool, default=True)
    parser.add_argument('--noise_schedule', type=str, default='linear') # linear or cosine
    parser.add_argument('--learn_sigma', type=bool, default=False)
    parser.add_argument('--z_sampling', type=bool, default=False) # zigzag samplingを行うかどうか
    parser.add_argument('--lambda_step', type=int, default=0)
    
    # Rectified Flowの設定
    parser.add_argument('--euler_step', type=int, default=3)
    parser.add_argument('--time_sampler', type=str, default='uniform') # uniform or logit_norm or exponential
    parser.add_argument('--euler_ensemble', type=int, default=0)
    parser.add_argument('--sampler_type', type=str, default='euler') # euler or ensemble_euler

    # 画像の詳細設定
    parser.add_argument('--img_size', type=int, default=256)    # 画像サイズ
    parser.add_argument('--mask_channels', type=int, default=1) # セグメンテーションマップのチャネル数
    parser.add_argument('--img_channels', type=int, default=3)  # 条件(医用)画像のチャネル数
    
    # 潜在空間の詳細設定
    parser.add_argument('--latent_size', type=int, default=32)  # 潜在変数のサイズ
    parser.add_argument('--latent_dim', type=int, default=4)    # 潜在変数のチャネル数
    
    # VAEの設定
    parser.add_argument('--vae_in_out_ch', type=int, default=1) # maskチャネルのサイズ
    parser.add_argument('--vae_mode', type=str, default='in_out_layer', choices=['adapter', 'in_out_layer']) # adapter or in_out_layer
    parser.add_argument('--vae_checkpoint', type=bool, default=True)
    parser.add_argument('--vae_path', type=str, default="/root/volume/model/staibility_vae_mask.pth") # /root/volume/model/staibility_vae_mask.pth, /root/volume/log/staible_vae_tanh_epochs_1000:REFUGE-Cup/weights/weight_epoch_700.pth
    parser.add_argument('--use_stability_vae', type=bool, default=True) # True: stability_ai, False: original
    parser.add_argument('--use_tanh', type=bool, default=False)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=1)
    parser.add_argument('--scaling_factor', type=float, default=0.18215) # 0.18215

    parser.add_argument('--train_size', type=int, default=0)
    parser.add_argument('--val_size', type=int, default=0)
    parser.add_argument('--test_size', type=int, default=0)

    parser.add_argument('--clip_grad', type=bool, default=False)
    parser.add_argument('--multi_gpu', type=bool, default=False) # pythonのバージョン3.9以下はFalse
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--transoform_version', type=str, default='v1')

    args = parser.parse_args()
    main(args)