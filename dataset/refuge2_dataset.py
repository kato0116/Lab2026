import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# one-hotに変換
def fundus_map_mask(mask, exclusive=False):
    num_classes = 3
    nhot_shape = list(mask.shape)
    if len(nhot_shape) == 2:
        nhot_shape.insert(0, num_classes)
    nhot_shape[-3] = num_classes
    
    if type(mask) == torch.Tensor:
        mask_nhot = torch.zeros(nhot_shape, device=mask.device)
    else:
        mask_nhot = np.zeros(nhot_shape)
    
    # Has the batch dimension
    if mask.ndim == 4:
        # Fake mask. No groundtruth mask available.
        if mask.shape[1] == 1:
            return mask_nhot

        mask_nhot[:, 0] = (mask[:, 0] == 0)     # 0        in channel 0 is background.
        if not exclusive:
            mask_nhot[:, 1] = (mask[:, 0] >= 1)                     # 1 or 255 in channel 0 is optic disc AND optic cup.
        else:
            mask_nhot[:, 1] = (mask[:, 0] >= 1) & (mask[:, 1] == 0) # 1 or 255 in channel 0, excluding 1/255 in channel 1 is optic disc only.
        mask_nhot[:, 2] = (mask[:, 1] >= 1)     # 1 or 255 in channel 1 is optic cup.
    # No batch dimension
    elif mask.ndim == 3:
        # Fake mask. No groundtruth mask available.
        if mask.shape[0] == 1:
            return mask_nhot

        mask_nhot[0] = (mask[0] == 0)           # 0        in channel 0 is background.
        if not exclusive:
            mask_nhot[1] = (mask[0] >= 1)                   # 1 or 255 in channel 0 is optic disc AND optic cup.
        else:
            mask_nhot[1] = (mask[0] >= 1) & (mask[1] == 0)  # 1 or 255 in channel 0, excluding 1/255 in channel 1 is optic disc only.
        mask_nhot[2] = (mask[1] >= 1)           # 1 or 255 in channel 1 is optic cup.
    # Convert REFUGE official annotation format to onehot encoding.
    elif mask.ndim == 2:
        # Fake mask. No groundtruth mask available.
        if mask.shape[0] == 1:
            return mask_nhot

        mask_nhot[0] = (mask == 255)                    # 255 (white) in channel 0 is background.
        if not exclusive:
            mask_nhot[1] = (mask <= 128)                # 128 or 0 is optic disc AND optic cup.
        else:
            mask_nhot[1] = mask[0] == 128               # 128 is optic disc only.
        mask_nhot[2] = (mask == 0)                      # 0 is optic cup.

    return mask_nhot

# one-hotを元に戻す
# def fundus_inv_map_mask(mask_nhot):
#     num_classes = 3
#     # original mask = 255: background
#     # original mask = 128: optic disc (including optic cup)
#     # original mask = 0:   optic cup
    
#     # Has the batch dimension
#     if len(mask_nhot.shape) == 4:
#         if type(mask_nhot) == np.ndarray:
#             mask = np.zeros_like(mask_nhot[:, 0],    dtype=np.uint8)
#         else:
#             mask = torch.zeros_like(mask_nhot[:, 0], dtype=torch.uint8)
#         mask[ mask_nhot[:, 0] == 1 ] = 255
#         mask[ mask_nhot[:, 1] == 1 ] = 128
#         mask[ mask_nhot[:, 2] == 1 ] = 0
#     # Single image, no batch dimension
#     elif len(mask_nhot.shape) == 3:
#         if type(mask_nhot) == np.ndarray:
#             mask = np.zeros_like(mask_nhot[0],    dtype=np.uint8)
#         else:
#             mask = torch.zeros_like(mask_nhot[0], dtype=torch.uint8)
#         mask[ mask_nhot[0] == 1 ] = 255
#         mask[ mask_nhot[1] == 1 ] = 128
#         mask[ mask_nhot[2] == 1 ] = 0
#     else:
#         breakpoint()
    
#     return mask
def fundus_inv_map_mask(mask_nhot):
    # original mask = 255: background
    # original mask = 128: optic disc (including optic cup)
    # original mask = 0:   optic cup

    if len(mask_nhot.shape) == 4:  # batch
        if isinstance(mask_nhot, np.ndarray):
            mask = np.full_like(mask_nhot[:, 0], 255, dtype=np.uint8)  # 初期値を255
        else:
            mask = torch.full_like(mask_nhot[:, 0], 255, dtype=torch.uint8)
        mask[mask_nhot[:, 1] == 1] = 128  # Disc
        mask[mask_nhot[:, 2] == 1] = 0    # Cup
    elif len(mask_nhot.shape) == 3:  # single image
        if isinstance(mask_nhot, np.ndarray):
            mask = np.full_like(mask_nhot[0], 255, dtype=np.uint8)
        else:
            mask = torch.full_like(mask_nhot[0], 255, dtype=torch.uint8)
        mask[mask_nhot[1] == 1] = 128
        mask[mask_nhot[2] == 1] = 0
    else:
        breakpoint()

    return mask

class REFUGEDataset(Dataset):
    def __init__(self, root_dir, mode="Train", transform=False, crop=False, center=False, split_ODOC=None): # mode: Train or Validation or Test
        self.subdir_path = os.path.join(root_dir, mode)
        mask_subdir_path = os.path.join(self.subdir_path, "Disc_Mask")
        self.name_list   = [f.path.split('/')[-1][:-4] for f in os.scandir(mask_subdir_path) if f.is_file()]
        self.crop   = crop
        self.center = center

        self.mode     = mode # Train or Val or Test
        self.transform = transform
        self.num_samples = len(self.name_list)
        self.split_ODOC = split_ODOC
    def __getitem__(self, index):
        name = self.name_list[index]
        if self.crop:
            # raw image and raters path
            if self.center:
                img_path     = os.path.join(self.subdir_path, "Crop_Images", name + ".jpg")
                if self.mode == "Train":
                    mask_path = os.path.join(self.subdir_path, "Crop_Mask", name + ".bmp")
                else:
                    mask_path = os.path.join(self.subdir_path, "Crop_Mask", name + ".png")
            else:
                img_path      = os.path.join(self.subdir_path, "Crop_Images2", name + ".jpg")
                if self.mode == "Train":
                    mask_path = os.path.join(self.subdir_path, "Crop_Mask2", name + ".bmp")
                else:
                    mask_path = os.path.join(self.subdir_path, "Crop_Mask2", name + ".png")
                
        else:
            # raw image and raters path
            img_path              = os.path.join(self.subdir_path, "Images", name + ".jpg")
            if self.mode == "Train":
                mask_path = os.path.join(self.subdir_path, "Disc_Mask", name + ".bmp")
            else:
                mask_path = os.path.join(self.subdir_path, "Disc_Mask", name + ".png")
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        np_mask = np.array(mask)
        # one-hot encoding
        nhot_mask = fundus_map_mask(np_mask, exclusive=False)
        mask = torch.tensor(nhot_mask, dtype=torch.float32)
        if self.split_ODOC == "optic_disc":
            mask = mask[1]
            mask = mask.repeat(3,1,1)
            
        elif self.split_ODOC == "optic_cup":
            mask = mask[2]
            mask = mask.repeat(3,1,1)
        
        
        if self.transform:
            state = torch.get_rng_state()
            img   = self.transform["image"](img)
            torch.set_rng_state(state)
            mask  = self.transform["mask"](mask)
        return (img, mask)

    def __len__(self):
        return self.num_samples
    
    
# from torchvision import transforms
# from torchvision.transforms import InterpolationMode

# transform = {
#     "image": transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]),
#             "mask": transforms.Compose([
#                 transforms.Resize(size=(1024,1024),interpolation=InterpolationMode.NEAREST),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 # transforms.RandomVerticalFlip(p=0.5),
#                 # transforms.RandomRotation(degrees=(0, 360)),
#                 # transforms.RandomAffine(
#                 #     degrees=45,                   # 回転範囲 (-45, 45)
#                 #     translate=(0.0625, 0.0625),   # 平行移動範囲 (-6.25%, +6.25%)
#                 #     scale=(0.9, 1.1),             # 拡大縮小範囲 (90% - 110%)
#                 # ),
#                 # transforms.ToTensor(),
#             ])
#     }

# dir_path = "/root/volume/dataset/REFUGE2"
# mode     = "Train"
# dataset = REFUGEDataset(dir_path, mode=mode, transform=transform)
# img, mask = dataset[0]
# print(np.unique(np.array(mask[2])))
# inv_mask = fundus_inv_map_mask(mask)
# from matplotlib import pyplot as plt
# plt.imshow(inv_mask, cmap='gray')
# # plt.axis('off')
# plt.savefig("./test.png")
# path = "/root/volume/dataset/REFUGE2"
# mode = "Train"
# dir_path = os.path.join(path, mode)
# subdir_path = os.path.join(dir_path, "Disc_Mask")
# # 拡張子を除いたファイル名のリストを取得 .bmp, .png, .jpg
# name_list = [f.path.split('/')[-1][:-4] for f in os.scandir(subdir_path) if f.is_file()]
# # name_list = [f.path.split('/')[-1] for f in os.scandir(subdir_path) if f.is_file()]
# print(name_list[0])


# mask_path = "/root/volume/dataset/REFUGE2/Test/Disc_Mask/T0010.png"
# img = Image.open(mask_path).convert('L')

# # ユニークな画素値を取得

# print("ユニークな画素値:", unique_values)
# np_mask = np.array(img)
# mask_nhot = fundus_map_mask(np_mask, exclusive=False)
# print(mask_nhot.shape)
# print(np.unique(mask_nhot[0]))
# mask_inv = fundus_inv_map_mask(mask_nhot)
# print(np.unique(mask_inv))
# print(mask_inv.shape)
# 画像だけでいい、横縦軸いらない
# plt.axis('off')



# plt.imshow(mask_nhot[0], cmap='gray')
# plt.savefig("./test.png")
# plt.imshow(mask_nhot[1], cmap='gray')
# plt.savefig("./test2.png")
# plt.imshow(mask_nhot[2],cmap='gray')
# plt.savefig("./test3.png")
# rgb_mask = np.zeros((mask_nhot.shape[1], mask_nhot.shape[2], 3), dtype=np.uint8)
# rgb_mask[mask_nhot[0] == 1] = [255, 255, 255]  # Background
# rgb_mask[mask_nhot[1] == 1] = [255, 0, 0]    # Optic disc
# rgb_mask[mask_nhot[2] == 1] = [0, 255, 0]    # Optic cup
# plt.imshow(rgb_mask, cmap='gray')
# plt.savefig("./test4.png")
# inv_mask = fundus_inv_map_mask(mask_nhot)
# plt.imshow(inv_mask, cmap='gray')
# plt.savefig("./test5.png")


# # 新しくcsvファイルを作成
# df = pd.DataFrame(columns=["path"])
# dir_path = "/root/volume/dataset/REFUGE2/Train/REFUGE1-train/Training400/Glaucoma"
# list_path = []
# for subfolder in os.listdir(dir_path):
#     list_path.append(subfolder)
# # csvファイルにlist_pathを追加  
# df["path"] = list_path
# # csvファイルを保存
# csv_path = "/root/volume/dataset/REFUGE2/Train/refuge1_train_label.csv"
# df.to_csv(csv_path, index=False)
