import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim,dropout=0, useconv=False):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        
        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        elif useconv:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_ch, out_ch, 3, padding=1)
            ),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_ch),
        )

        self.time_emb = nn.Linear(time_embed_dim, 2*out_ch)

    def forward(self, x, v):
        h = self.in_layer(x)
        if v is not None:
            emb_out = self.time_emb(v)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            
            out_norm, out_rest = self.out_layer[0], self.out_layer[1:]
            shift, scale = emb_out.chunk(2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layer(h)
        return h+self.skip_connection(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.down  = nn.MaxPool2d(2)
        self.convs = ResBlock(in_ch, out_ch, time_embed_dim)
    
    def forward(self, x, v):
        x = self.down(x)
        x = self.convs(x, v)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = ResBlock(in_ch, out_ch, time_embed_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
    
    def forward(self, x, v):
        x = self.convs(x, v)
        x = self.up(x)
        return x
    
class UnetEnc(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, hiddn_dim=128):
        super().__init__()
        self.conv_in  = nn.Conv2d(in_ch, hiddn_dim, 3, padding=1)
        self.in_layer = ResBlock(hiddn_dim, hiddn_dim, time_embed_dim)       
        self.down1 = DownBlock(hiddn_dim, hiddn_dim, time_embed_dim)
        self.down2 = DownBlock(hiddn_dim, hiddn_dim, time_embed_dim)
        self.down3 = DownBlock(hiddn_dim, hiddn_dim, time_embed_dim)
        self.down4 = DownBlock(hiddn_dim, hiddn_dim*2, time_embed_dim)
        self.down5 = DownBlock(hiddn_dim*2, hiddn_dim*4, time_embed_dim)
        self.down6 = DownBlock(hiddn_dim*4, hiddn_dim*8, time_embed_dim)
    def forward(self, x, v=None):
        x  = self.conv_in(x) 
        x0 = self.in_layer(x, v) 
        x1 = self.down1(x0, v)
        x2 = self.down2(x1, v)
        x3 = self.down3(x2, v)
        x4 = self.down4(x3, v)
        x5 = self.down5(x4, v)
        x6 = self.down6(x5, v)
        return  [x3, x4, x5, x6]

class LatentUnetEnc(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, hiddn_dim=32):
        super().__init__()
        # self.conv_in = TwoConvBlock(in_ch, hiddn_dim, time_embed_dim)
        self.conv_in = nn.Conv2d(in_ch, hiddn_dim, 3, padding=1)
        self.in_layer = ResBlock(hiddn_dim, hiddn_dim, time_embed_dim)       
        self.down1 = DownBlock(hiddn_dim, hiddn_dim*2, time_embed_dim)
        self.down2 = DownBlock(hiddn_dim*2, hiddn_dim*4, time_embed_dim)
        self.down3 = DownBlock(hiddn_dim*4, hiddn_dim*8, time_embed_dim)
                
    def forward(self, x, y_list, v):
        x = self.conv_in(x)      # 32x32
        x0 = self.in_layer(x, v) # 32x32
        x0 = x0+y_list[0]
        x1 = self.down1(x0, v)   # 16x16
        x1 = x1+y_list[1]
        x2 = self.down2(x1, v)   # 8x8
        x2 = x2+y_list[2]
        x = self.down3(x2, v)   # 4x4
        x = x+y_list[3]
        skip = [x0, x1, x2]
        return  x, skip
    
class LatentUnetDec(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, hiddn_dim=32):
        super().__init__()
        # self.up1 = UpBlock(hiddn_dim*16, hiddn_dim*8, time_embed_dim)
        self.up2 = UpBlock(hiddn_dim*8, hiddn_dim*4, time_embed_dim)
        self.up3 = UpBlock(hiddn_dim*8, hiddn_dim*2, time_embed_dim)
        self.up4 = UpBlock(hiddn_dim*4, hiddn_dim, time_embed_dim)
        self.conv_out = ResBlock(hiddn_dim*2, hiddn_dim, time_embed_dim)
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, hiddn_dim),
            nn.SiLU(),
            nn.Conv2d(hiddn_dim, out_ch, 3, padding=1),
        )
    def forward(self, x, skip, v):
        x = self.up2(x, v)
        x = torch.cat([x, skip.pop(-1)], dim=1)
        x = self.up3(x, v)
        x = torch.cat([x, skip.pop(-1)], dim=1)
        x = self.up4(x, v)
        x = torch.cat([x, skip.pop(-1)], dim=1)
        x = self.conv_out(x,v)
        x = self.out_layer(x)
        
        return x

class LSegUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        cond_in_channels=3,
        unet_hidden_size=128,
        time_embed_dim=256,
    ):
        super().__init__()
        self.t_embedder    = TimestepEmbedder(time_embed_dim)
        self.image_encoder = UnetEnc(cond_in_channels, in_channels, time_embed_dim, unet_hidden_size)
        self.encoder = LatentUnetEnc(in_channels, in_channels, time_embed_dim, unet_hidden_size)
        self.decoder = LatentUnetDec(in_channels, in_channels, time_embed_dim, unet_hidden_size)
        self.bottle  = nn.ModuleList([
            ResBlock(unet_hidden_size*8, unet_hidden_size*8, time_embed_dim) for _ in range(2)
        ])
        
    def forward(self, x, v, conditioned_image):
        v = self.t_embedder(v)
        y_list = self.image_encoder(conditioned_image)
        x, skip = self.encoder(x, y_list, v)
        for block in self.bottle:
            x = block(x, v)
        x = self.decoder(x, skip, v)
        return x