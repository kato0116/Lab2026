import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
class BratsDataset(Dataset):
    def __init__(self,df,transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        mask_path = self.df["seg_path"].iloc[idx]
        img_path  = self.df["t1c_path"].iloc[idx]
        # OpenCVではなくPILを使用して画像を読み込む
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path)

        # img  = img.transpose(2,0,1)
        # mask = mask.transpose(2,0,1)
        # img  = torch.tensor(img[0])
        # img  = img.unsqueeze(0)
        # mask = torch.tensor(mask[0])

        # et  = (mask == 85).type(torch.float32)          # Enhance Tumor
        # tc  = ((mask == 255) | (mask==85)).type(torch.float32) # Tumor Core
        # wt  = ((mask == 255) | (mask==85) | (mask==170)).type(torch.float32) # Whole Tumor
        # mask = torch.stack([et,tc,wt],dim=0)
        # print(mask.shape)
        # print(img.shape)
        if self.transform:
            img  = self.transform(img)
            mask = self.transform(mask)

        return img,mask

# csv_path = "/root/volume/dataset/Brats2023/df.csv"
# df = pd.read_csv(csv_path)
# # df.info()
# # # diagnosisの割り当て

# df = df[df["diagnosis"]==1] # 陽性のみ
# transform = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor()
# ])
# dataset = BratsDataset(df,transform=transform)
# print(df["diagnosis"].value_counts())
# i = 0
# for img,mask in dataset:
#     i +=1
#     if i == 90:
#         break
# y_list = []
# for x in mask[0]:
#     for y in x:
#         if y not in y_list:
#             y_list.append(y)
# print(y_list)
# print(mask.shape)

# grid = torchvision.utils.make_grid(mask)
# plt.imshow(grid.numpy().transpose((1,2,0)))
# plt.show()
# plt.savefig("./mask.png")

# loader = DataLoader(dataset,batch_size=32,shuffle=True)
# for img,mask in loader:
#     print(img.shape)
#     print(mask.shape)
#     break
    