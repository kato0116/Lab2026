import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch as th
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import torch.nn as nn
from guided_diffusion.dist_util import cleanup, init_process
from utils.utils import requires_grad, update_ema, calc_metric
from utils.strage import save_model, load_model
from utils.staple import staple
from metric.metric import dice, jaccard, sensitivity, specificity, accuracy
from skimage.filters import threshold_otsu
from copy import deepcopy
import wandb
from diffusers.models import AutoencoderKL
import torch.nn.functional as F
import numpy as np
import random
from dataset.refuge2_dataset import fundus_inv_map_mask
from utils.create_vae import create_vae

class Trainer:
    def __init__(
        self,
        *,
        model,
        diffuser,
        optimizer,
        train_set,
        val_set,
        test_set,
        args,
        dir_path,
        loss_function,
        scheduler=None,
        space="latent",
        parameter_num=None,
        trainable_param=None,
        multi_gpu=False,
    ):
        self.multi_gpu     = multi_gpu
        if multi_gpu:
            self.rank, self.device, self.seed = init_process(args)
            self.model = DDP(
                model.to(self.device),
                device_ids=[self.rank],
            )
            self.train_sampler, self.train_loader = self.get_loader(dataset=train_set, shuffle=True,  seed=args.global_seed, batch_size=args.global_batch_size, num_workers=args.num_workers, drop_lat=True)
            self.val_sampler, self.val_loader     = self.get_loader(dataset=val_set,   shuffle=False, seed=args.global_seed, batch_size=args.microbatch, num_workers=args.num_workers, drop_lat=False)
            self.test_sampler, self.test_loader   = self.get_loader(dataset=test_set,  shuffle=False, seed=args.global_seed, batch_size=1, num_workers=args.num_workers, drop_lat=False)
        else:
            self.rank         = 0
            self.device       = th.device("cuda" if th.cuda.is_available() else "cpu")
            self.model        = model.to(self.device)
            self.train_loader = DataLoader(train_set, batch_size=args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            self.val_loader   = DataLoader(val_set, batch_size=args.microbatch, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
            self.test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)


        self.use_ema           = args.use_ema
        self.diffuser_type     = args.diffuser_type
        self.checkpoint        = args.checkpoint
        self.checkpoint_epoch  = 0                  # args.checkpoint_epoch
        self.epochs            = args.epochs
        self.microbatch        = args.microbatch
        
        self.train_size        = args.train_size
        self.val_size          = args.val_size
        self.test_size         = args.test_size
        
        self.val_ensemble      = args.val_ensemble
        self.test_ensemble     = args.test_ensemble
        
        self.ch3_to_ch1        = args.ch3_to_ch1
        self.val_step_num      = args.val_step_num
        self.wandb_num_images  = args.wandb_num_images
        
        self.optimizer         = optimizer
        self.scheduler         = scheduler
        self.parameter_num     = parameter_num
        self.trainable_param   = trainable_param
        self.dir_path          = dir_path
        self.space             = space
        self.scaling_factor    = args.scaling_factor
        
        self.staple_flag       = args.staple_flag
        self.dataset           = args.dataset
        self.split_ODOC        = args.split_ODOC
        
        self.clip_grad         = args.clip_grad

        self.best_path            = os.path.join(dir_path,"weights",f"weight_epoch_best_dice.pth")
        self.ema_best_path        = os.path.join(dir_path,"weights",f"weight_epoch_best_dice_ema.pth")
        self.best_staple_path     = os.path.join(dir_path,"weights",f"weight_epoch_best_staple_dice.pth")
        self.ema_best_staple_path = os.path.join(dir_path,"weights",f"weight_epoch_best_staple_dice_ema.pth")
        self.last_path            = os.path.join(dir_path,"weights",f"weight_epoch_last.pth")
        self.ema_last_path        = os.path.join(dir_path,"weights",f"weight_epoch_last_ema.pth")
        self.loss_function        = loss_function
        self.best_dice            = 0.0
        self.best_staple_dice     = 0.0

        # EMAモデルの用意
        if self.use_ema:
            print("Using EMA Model...")
            self.ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
            requires_grad(self.ema, False)

        
        self.image_shape  = (args.img_channels, args.img_size, args.img_size)
        self.mask_shape   = (args.mask_channels, args.img_size, args.img_size)
        print(f"Image shape: {self.image_shape}, Mask shape: {self.mask_shape}")
    
    def train_loop(self, args):
        if self.use_ema:
            if self.checkpoint:
                print("Reading the EMA checkpoint...")
                self.ema = load_model(self.ema, self.ema_checkpoint_path).to(self.device)
                requires_grad(self.ema, False)
            else:
                update_ema(self.ema, self.model, decay=0)
            self.ema.eval()

        if self.rank == 0:
            self.init_wandb(args, train_flag=True)
            
        for epoch in range(self.checkpoint_epoch+1, self.epochs + 1):
            if self.rank == 0:
                print(f"epoch:{epoch}/{self.epochs}")
                train_losses       = []
                
            for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.rank != 0):
                self.model.train()
                image, mask = batch
                self.optimizer.zero_grad()
                for i in range(0, mask.shape[0],self.microbatch):
                    micro_mask   = mask[i:i+self.microbatch]
                    micro_image  = image[i:i+self.microbatch]
                    micro_mask   = micro_mask.to(self.device)
                    x            = micro_mask.to(self.device)
                    y            = micro_image.to(self.device)
                    
                    pred_x = self.model(y)
                    loss = self.loss_function(pred_x, x)
                    loss.backward()
                    if self.rank == 0:
                        train_losses.append(loss.item()*len(x))
                if self.rank == 0 and self.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                if self.use_ema:
                    update_ema(self.ema, self.model)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.multi_gpu:
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)
                dist.barrier()
            
            train_avg_loss = sum(train_losses) / self.train_size
            
            wandb_lr = self.optimizer.param_groups[0]['lr']
            print(f"train_loss:{train_avg_loss}, not send image")
            if epoch % self.val_step_num == 0:
                log_config = self.val_loop(phase="val")
            else:
                log_config = {}
            log_config.update({
                "train_loss": train_avg_loss,
                "lr": wandb_lr
            })
            wandb.log({
                **log_config
            })
        
        # 最終エポックのモデルを保存
        if self.rank == 0:
            save_model(self.model, "last", self.dir_path)
            if self.use_ema:
                save_model(self.ema, "last_ema", self.dir_path)
            wandb.finish()  
    
    def val_loop(self, phase="val"):
        val_disc_dice_otsu = []
        val_disc_iou_otsu  = []
        val_disc_hausdorff_otsu = []
        val_disc_hausdorff95_otsu = []
        val_cup_dice_otsu  = []
        val_cup_iou_otsu   = []
        val_cup_hausdorff_otsu = []
        val_cup_hausdorff95_otsu = []
        val_dice_otsu      = []
        val_iou_otsu       = []
        val_hausdorff_otsu = []
        val_hausdorff95_otsu = []

        if phase == "val":
            data_loader  = self.val_loader
            ensemble_num = self.val_ensemble
            data_size    = self.val_size
        elif phase == "test":
            data_loader = self.test_loader
            ensemble_num = self.test_ensemble
            data_size    = self.test_size
        else:
            raise ValueError("mode should be 'val' or 'test'")
        
        if self.use_ema:
            self.ema.eval()
        self.model.eval()
        for image, mask in tqdm(data_loader):
            x_start = mask.to(self.device)
            y       = image.to(self.device)
            self.model.eval()
            with th.no_grad():
                pred_x = self.model(y)

            if self.dataset == "REFUGE2" or self.dataset == "REFUGE2_Crop" or self.dataset == "REFUGE1" or self.dataset == "REFUGE1_Crop":
                if self.split_ODOC == "optic_cup" or self.split_ODOC == "optic_disc":
                    # 3チャネルを平均
                    pred_x = th.mean(pred_x, dim=1)
                    # 閾値処理
                    threshold                = threshold_otsu(pred_x.cpu().numpy())
                    binary_pred_x = (pred_x > threshold).float()
                else:
                    # meanを使う場合
                    pred_x0 = pred_x[:,0]
                    pred_x1 = pred_x[:,1]
                    pred_x2 = pred_x[:,2]
                    th0 = threshold_otsu(pred_x0.cpu().numpy())
                    th1 = threshold_otsu(pred_x1.cpu().numpy())
                    th2 = threshold_otsu(pred_x2.cpu().numpy())
                    binary_pred_x0 = (pred_x0 > th0).float()
                    binary_pred_x1 = (pred_x1 > th1).float()
                    binary_pred_x2 = (pred_x2 > th2).float()
                    binary_pred_x = th.stack([binary_pred_x0, binary_pred_x1, binary_pred_x2], dim=1)
            else:
                threshold = threshold_otsu(pred_x.cpu().numpy())
                binary_pred_x = (pred_x > threshold).float()
            
            if self.rank == 0:
                if self.dataset == "REFUGE2" or self.dataset == "REFUGE2_Crop" or self.dataset == "REFUGE1" or self.dataset == "REFUGE1_Crop":
                    binary_pred_x1 = binary_pred_x1.cpu().numpy()
                    binary_pred_x2 = binary_pred_x2.cpu().numpy()
                    mask = mask.cpu().numpy()
                    binary_pred_x0 = binary_pred_x0.cpu().numpy()

                    th1_otsu_metric = calc_metric(binary_pred_x1, mask[:,1])
                    th2_otsu_metric = calc_metric(binary_pred_x2, mask[:,2])
                                
                    val_disc_dice_otsu.append(th1_otsu_metric["dice"])
                    val_disc_iou_otsu.append(th1_otsu_metric["iou"])
                    val_disc_hausdorff_otsu.append(th1_otsu_metric["hausdorff"])
                    val_disc_hausdorff95_otsu.append(th1_otsu_metric["hausdorff_95"])
                    
                    val_cup_dice_otsu.append(th2_otsu_metric["dice"])
                    val_cup_iou_otsu.append(th2_otsu_metric["iou"])
                    val_cup_hausdorff_otsu.append(th2_otsu_metric["hausdorff"])
                    val_cup_hausdorff95_otsu.append(th2_otsu_metric["hausdorff_95"])

                    mask = fundus_inv_map_mask(mask)
                    binary_pred_x = fundus_inv_map_mask(binary_pred_x)
            
                else:
                    binary_pred_x   = binary_pred_x.cpu().numpy()
                    if self.ch3_to_ch1:
                        mask = mask[:,0,:,:].unsqueeze(1).cpu().numpy()
                    else:
                        mask = mask.cpu().numpy()                        
                    
                    th_otsu_metric = calc_metric(binary_pred_x, mask)
                    val_dice_otsu.append(th_otsu_metric["dice"])
                    val_iou_otsu.append(th_otsu_metric["iou"])
                    val_hausdorff_otsu.append(th_otsu_metric["hausdorff"])
                    val_hausdorff95_otsu.append(th_otsu_metric["hausdorff_95"])
        
        if self.rank == 0:        
            wandb_num_images  = min(self.wandb_num_images, y.shape[0], x_start.shape[0], binary_pred_x.shape[0])
            wandb_y           = [wandb.Image(y[i]) for i in range(wandb_num_images)]
            wandb_x           = [wandb.Image(mask[i]) for i in range(wandb_num_images)]
            wandb_pred_x_otsu = [wandb.Image(binary_pred_x[i]) for i in range(wandb_num_images)]
            # print(y.min(), y.max())
            # print("最小値、最大値")

            print("Saved the first image of y as test000.png for visualization.")
            if self.dataset == "REFUGE2" or self.dataset == "REFUGE2_Crop" or self.dataset == "REFUGE1" or self.dataset == "REFUGE1_Crop":
                avg_metric = {
                    "disc_dice": sum(val_disc_dice_otsu) / data_size,
                    "disc_iou":  sum(val_disc_iou_otsu)  / data_size,
                    "cup_dice":  sum(val_cup_dice_otsu)  / data_size,
                    "cup_iou":   sum(val_cup_iou_otsu)   / data_size,
                    "disc_hausdorff": sum(val_disc_hausdorff_otsu) / data_size,
                    "disc_hausdorff95": sum(val_disc_hausdorff95_otsu) / data_size,
                    "cup_hausdorff": sum(val_cup_hausdorff_otsu) / data_size,
                    "cup_hausdorff95": sum(val_cup_hausdorff95_otsu) / data_size,
                }
                print(f"best_cup_dice:{self.best_dice}")
                print(f"now_cup_dice:{avg_metric['cup_dice']}")
                if self.best_dice < (avg_metric["cup_dice"]+avg_metric["disc_dice"])/2:
                    self.best_dice = (avg_metric["cup_dice"]+avg_metric["disc_dice"])/2
                    print(f"best_dice:{self.best_dice}更新")
                    save_model(self.model, "best_dice", self.dir_path)
                    if self.use_ema:
                        save_model(self.ema, "best_dice_ema", self.dir_path)
                
            else:
                avg_metric = {
                    "dice": sum(val_dice_otsu) / data_size,
                    "iou":  sum(val_iou_otsu)  / data_size,
                    "hausdorff": sum(val_hausdorff_otsu) / data_size,
                    "hausdorff95": sum(val_hausdorff95_otsu) / data_size,
                }
                print(f"best_dice:{self.best_dice}")
                print(f"now_dice:{avg_metric['dice']}")
                if self.best_dice < avg_metric["dice"]:
                    self.best_dice = avg_metric["dice"]
                    print(f"best_dice:{self.best_dice}更新")
                    save_model(self.model, "best_dice", self.dir_path)
                    if self.use_ema:
                        save_model(self.ema, "best_dice_ema", self.dir_path)
                
            log_dict = {
                "otsu": avg_metric,
                "y": wandb_y,
                "x": wandb_x,
                "pred_x_otsu": wandb_pred_x_otsu,
            }

            return log_dict

    def test_loop(self, model, args):
        test_dice_list        = []
        test_jaccad_list      = []
        test_hausdorff_list   = []
        test_hausdorff95_list = []
        
        test_disc_dice_list   = []
        test_disc_jaccad_list = []
        test_disc_hausdorff_list   = []
        test_disc_hausdorff95_list = []
        test_cup_dice_list    = []
        test_cup_jaccad_list  = []
        test_cup_hausdorff_list   = []
        test_cup_hausdorff95_list = []

        # best modelを読み込む
        if args.use_best_model:
            if self.staple_flag:
                if self.use_ema:
                    print("Reading the best EMA Staple Model...")
                    print(self.ema_best_staple_path)
                    path = self.ema_best_staple_path
                else:
                    print("Reading the best Staple model...")
                    print(self.best_staple_path)
                    path = self.best_staple_path
            else:
                if self.use_ema:
                    print("Reading the best EMA Model...")
                    print(self.ema_best_path)
                    path = self.best_path
                else:
                    print("Reading the best model...")
                    print(self.best_path)
                    path = self.best_path
        else:
            if self.use_ema:
                print("Reading the last EMA Model...")
                print(self.ema_last_path)
                path = self.ema_last_path
            else:
                print("Reading the last model...")
                print(self.last_path)
                path = self.last_path
        print("Model loaded from:", path)
        model = load_model(model, path)

        if self.multi_gpu:
            self.rank, self.device, self.seed = init_process(args)
            self.model = DDP(
                model.to(self.device),
                device_ids=[self.rank],
            )
        else:
            self.rank = 0
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
            self.model  = model.to(self.device)

        # wandbの初期化
        self.init_wandb(args, train_flag=False)
        print("Evaluating on test dataset...")
        self.model.eval()
        for image, mask in tqdm(self.test_loader):
            x_start = mask.to(self.device)
            y       = image.to(self.device)
            with th.no_grad():
                pred_x = self.model(y)

            if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                    # 3チャネルを平均
                    pred_x = th.mean(pred_x, dim=1)
                    # 閾値処理
                    threshold                = threshold_otsu(pred_x.cpu().numpy())
                    binary_pred_x = (pred_x > threshold).float()
                else:
                    pred_x0 = pred_x[:,0]
                    pred_x1 = pred_x[:,1]
                    pred_x2 = pred_x[:,2]
                    th0 = threshold_otsu(pred_x0.cpu().numpy())
                    th1 = threshold_otsu(pred_x1.cpu().numpy())
                    th2 = threshold_otsu(pred_x2.cpu().numpy())
                    binary_pred_x0 = (pred_x0 > th0).float()
                    binary_pred_x1 = (pred_x1 > th1).float()
                    binary_pred_x2 = (pred_x2 > th2).float()
                    binary_pred_x = th.stack([binary_pred_x0, binary_pred_x1, binary_pred_x2], dim=1)

            else:
                threshold = threshold_otsu(pred_x.cpu().numpy())
                binary_pred_x = (pred_x > threshold).float()

            if self.rank == 0:
                if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                    if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                        binary_pred_x = binary_pred_x.unsqueeze(1).cpu().numpy()
                        mask = mask[:,0].unsqueeze(1).cpu().numpy()
                        th_otsu_metric = calc_metric(binary_pred_x, mask)
                        test_dice_list.append(th_otsu_metric["dice"])
                        test_jaccad_list.append(th_otsu_metric["iou"])
                    else:
                        binary_pred_x1 = binary_pred_x1.cpu().numpy()
                        binary_pred_x2 = binary_pred_x2.cpu().numpy()
                        mask = mask.cpu().numpy()
                        binary_pred_x0 = binary_pred_x0.cpu().numpy()

                        th1_otsu_metric = calc_metric(binary_pred_x1, mask[:,1])
                        th2_otsu_metric = calc_metric(binary_pred_x2, mask[:,2])
                        mask = fundus_inv_map_mask(mask)
                        binary_pred_x = fundus_inv_map_mask(binary_pred_x)
                        
                        test_disc_dice_list.append(th1_otsu_metric["dice"])
                        test_disc_jaccad_list.append(th1_otsu_metric["iou"])
                        test_disc_hausdorff_list.append(th1_otsu_metric["hausdorff"])
                        test_disc_hausdorff95_list.append(th1_otsu_metric["hausdorff_95"])
                        test_cup_dice_list.append(th2_otsu_metric["dice"])
                        test_cup_jaccad_list.append(th2_otsu_metric["iou"])
                        test_cup_hausdorff_list.append(th2_otsu_metric["hausdorff"])
                        test_cup_hausdorff95_list.append(th2_otsu_metric["hausdorff_95"])

                else:
                    binary_pred_x   = binary_pred_x.cpu().numpy()
                    if self.ch3_to_ch1:
                        mask = mask[:,0,:,:].unsqueeze(1).cpu().numpy()
                    else:
                        mask = mask.cpu().numpy()
            
                    th_otsu_metric = calc_metric(binary_pred_x, mask)
                    test_dice_list.append(th_otsu_metric["dice"])
                    test_jaccad_list.append(th_otsu_metric["iou"])
                    test_hausdorff_list.append(th_otsu_metric["hausdorff"])
                    test_hausdorff95_list.append(th_otsu_metric["hausdorff_95"])
            
            if self.rank == 0:
                wandb_num_images = min(self.wandb_num_images, y.shape[0], x_start.shape[0], binary_pred_x.shape[0])
                wandb_y = [wandb.Image(y[i].cpu()) for i in range(wandb_num_images)]
                wandb_x = [wandb.Image(mask[i]) for i in range(wandb_num_images)]
                wandb_pred_x_otsu = [wandb.Image(binary_pred_x[i]) for i in range(wandb_num_images)]
                
                log_config = {
                    "image": wandb_y,
                    "mask": wandb_x,
                    "pred_otsu_mask": wandb_pred_x_otsu,
                }
                if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                    log_config.update({
                        "disc_dice": th1_otsu_metric["dice"],
                        "disc_iou": th1_otsu_metric["iou"],
                        "cup_dice": th2_otsu_metric["dice"],
                        "cup_iou": th2_otsu_metric["iou"],
                    })
                else:  
                    log_config.update({
                        "dice": th_otsu_metric["dice"],
                        "iou": th_otsu_metric["iou"],
                    })

                wandb.log(log_config)
        
        
        if self.rank == 0:
            if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                test_disc_dice   = sum(test_disc_dice_list) / len(test_disc_dice_list)
                test_cup_dice    = sum(test_cup_dice_list) / len(test_cup_dice_list)
                test_disc_jaccad = sum(test_disc_jaccad_list) / len(test_disc_jaccad_list)
                test_cup_jaccad  = sum(test_cup_jaccad_list) / len(test_cup_jaccad_list)
                test_disc_hausdorff = sum(test_disc_hausdorff_list) / len(test_disc_hausdorff_list)
                test_disc_hausdorff95 = sum(test_disc_hausdorff95_list) / len(test_disc_hausdorff95_list)
                test_cup_hausdorff = sum(test_cup_hausdorff_list) / len(test_cup_hausdorff_list)
                test_cup_hausdorff95 = sum(test_cup_hausdorff95_list) / len(test_cup_hausdorff95_list)
                
                print(f"test_disc_dice:{test_disc_dice}")
                print(f"test_cup_dice:{test_cup_dice}")
                log_config =({
                        "test_disc_dice": test_disc_dice,
                        "test_cup_dice": test_cup_dice,
                        "test_disc_iou": test_disc_jaccad,
                        "test_cup_iou": test_cup_jaccad,
                        "test_disc_hausdorff": test_disc_hausdorff,
                        "test_disc_hausdorff95": test_disc_hausdorff95,
                        "test_cup_hausdorff": test_cup_hausdorff,
                        "test_cup_hausdorff95": test_cup_hausdorff95,
                    })
            else:
                test_dice        = sum(test_dice_list) / len(test_dice_list)
                test_jaccad      = sum(test_jaccad_list) / len(test_jaccad_list)
                test_hausdorff   = sum(test_hausdorff_list) / len(test_hausdorff_list)
                test_hausdorff95 = sum(test_hausdorff95_list) / len(test_hausdorff95_list)
            
                print(f"test_dice:{test_dice}")
                print(f"test_jaccad:{test_jaccad}")
                log_config = ({
                    "test_dice": test_dice,
                    "test_jaccad": test_jaccad,
                    "test_hausdorff": test_hausdorff,
                    "test_hausdorff95": test_hausdorff95,
                })
            wandb.log(log_config)
            wandb.finish()
            
        if self.multi_gpu:
            dist.barrier()
            cleanup()
            
    def init_wandb(self, args, path=None, train_flag=True):
        config = {
            "model":            args.model_name,
            "epochs":           args.epochs,
            "image_size":       args.img_size,
            "mask_channel":     args.mask_channels,
            "img_channel":      args.img_channels,
            
            "diffuser_type":    None,
            "seed":             args.seed, 
            
            "batch_size":       args.global_batch_size,
            "learning_rate":    args.lr,
            "clip_grad":        args.clip_grad,
            "use_ema":          args.use_ema,
                        
            "patch_size":       args.patch_size,
            "skip_flag":        args.skip_flag,
            "cross_attn_flag":  args.cross_attn_flag,
            "shared_step":      args.shared_step,
            
            "param_num":         self.parameter_num,
            "trainable_param":   self.trainable_param,
            "train_size":        args.train_size,
            "val_size":          args.val_size,
            "test_size":         args.test_size,
            "dir_path":          self.dir_path,
            
            "Mamba(scan_type)":         args.scan_type,
            "Mamba(expand)":            args.expand,
            "Mamba(d_state)":           args.d_state,
            "Mamba(version)":           2 if args.use_mamba2 else 1,
            "Mamba(num_block)":         args.num_blocks,
        }
        tags = [args.dataset, args.cond, self.space,args.model_size]

        if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1"  or args.dataset == "REFUGE1_Crop":
            name = args.model_name + "_"+str(args.img_size)
        if args.fold != "":
            name = args.model_name + "_fold:" + args.fold
            tags.append("fold:"+args.fold)
            config.update({
                "fold":             args.fold,
            })
        if train_flag:
            project_name = args.project_name
            config.update({
                "val_ensemble":     args.val_ensemble,
            })
            with open(os.path.join(self.dir_path,"train_config.txt"), "w") as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
        else:
            project_name = args.project_name + "_test"
            if args.use_best_model:
                name = name + "(best)"
            else:
                name = name + "(last)"
            config.update({
                # "test_ensemble":    args.test_ensemble,
                "model_path":       path,
            })
            with open(os.path.join(self.dir_path,"test_config.txt"), "w") as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
        
        if args.checkpoint:
            config.update({
                "checkpoint":       args.checkpoint,
                "checkpoint_epoch": args.checkpoint_epoch,
            })
        wandb.init(
            project=project_name,
            name=name,
            tags=tags,
            config=config,
        )  

    def send_wandb(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass