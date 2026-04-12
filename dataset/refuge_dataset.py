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
        self.crop = crop
        self.center  = center

        self.mode     = mode # Train or Val or Test
        self.transform = transform
        self.num_samples = len(self.name_list)
        self.split_ODOC = split_ODOC
    def __getitem__(self, index):
        name = self.name_list[index]
        if self.crop:
            # raw image and raters path
            if self.center:
                img_path  = os.path.join(self.subdir_path, "Crop_Images", name + ".jpg")
                mask_path = os.path.join(self.subdir_path, "Crop_Mask", name + ".bmp")
            else:
                img_path  = os.path.join(self.subdir_path, "Crop_Images2", name + ".jpg")
                mask_path = os.path.join(self.subdir_path, "Crop_Mask2", name + ".bmp")       
        else:
            # raw image and raters path
            img_path              = os.path.join(self.subdir_path, "Images", name + ".jpg")
            mask_path = os.path.join(self.subdir_path, "Disc_Mask", name + ".bmp")

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
    