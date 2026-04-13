
import torch
import numpy as np
import random
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # GPUの乱数も固定（必要なら）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # マルチGPU用
        torch.backends.cudnn.deterministic = True  # CuDNNの動作を固定
        torch.backends.cudnn.benchmark = False  # 再現性を優先

def set_dataset(args):
    if args.dataset == "REFUGE2_Crop":
        print("Using REFUGE2_Crop dataset")
        args.data_path = "/root/volume/dataset/REFUGE2_ver_crop_check"
        args.center    = True
        args.crop      = True
        args.fold      = ""
        args.mask_channels  = 3
        args.vae_in_out_ch  = 3
        args.vae_checkpoint = False
        args.vae_path       = None

    elif args.dataset == "REFUGE2":
        args.data_path = "/root/volume/dataset/REFUGE2"  
        args.fold = ""
        args.mask_channels  = 3
        args.vae_in_out_ch  = 3
        args.vae_checkpoint = False
        args.vae_path       = None
    return args

def set_name(args):
    
    if args.diffuser_type == None: # 拡散モデルではない場合
        pass
    elif args.diffuser_type == "diffusion":
        args.model_name = "Diffusion" + "_" + args.model_name + "_"+args.model_size
    elif args.diffuser_type == "rectified_flow":
        args.model_name = "RF" + "_" + args.model_name + "_"+args.model_size
    
    # REFUGE系統のデータセット
    if args.dataset in ["REFUGE2", "REFUGE2_Crop", "REFUGE1_Crop"]:
        if args.center:
            args.model_name = args.model_name + "(center)"
        dir_path = os.path.join(args.path,"log",str(args.img_size)+"_"+args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)    
    # ISIC系統のデータセット
    elif args.dataset in ["ISIC2018", "ISIC2017", "ISIC2016"]:
        if args.fold == "":
            dir_path = os.path.join(args.path,"log",args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
        else:
            dir_path = os.path.join(args.path,"log","fold:"+args.fold+"_"+args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
            
    else:
        dir_path = os.path.join(args.path,"log",args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
    return args, dir_path
        
        

