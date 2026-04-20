import torch
import math
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Mlp
from torch import nn
from models.modules import PatchMerging, PatchExpand
from models.mamba_utils import zigzag_path, reverse_permut_np, raster_path
from mamba_ssm import Mamba, Mamba2

def create_path(input_size, patch_size, scan_type="zigzagN8"):
    num_patches = (input_size // patch_size) ** 2
    patch_side_len = int(math.sqrt(num_patches))
        
    if scan_type.startswith("zigzagN") or scan_type.startswith("parallelN"):
        zigzag_num = int(scan_type.replace("zigzagN", ""))
        _zz_paths = zigzag_path(N=patch_side_len)
        if scan_type.startswith("zigzagN"):
            zigzag_num = int(scan_type.replace("zigzagN", ""))
            zz_paths = _zz_paths[:zigzag_num]
            assert (
                len(zz_paths) == zigzag_num
                ), f"{len(zz_paths)} != {zigzag_num}"
        elif scan_type.startswith("parallelN"):
            zz_paths  = _zz_paths[:8]

    elif  scan_type.startswith("rasterN"):
        zigzag_num = int(scan_type.replace("rasterN", ""))
        raster_num = int(scan_type.replace("rasterN", ""))
        _raster_paths = raster_path(N=patch_side_len, num_directions=raster_num)
        zz_paths = _raster_paths
    zz_paths     = zz_paths
    zz_paths_rev = [reverse_permut_np(_) for _ in zz_paths]

    return zz_paths, zz_paths_rev


class MambaBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0, skip=False, zz_paths=None,zz_paths_rev=None, use_mamba2=False,d_state=16,expand=1,layer_id=None, **block_kwargs):
        super().__init__()
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.norm1       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn        = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        if use_mamba2:
            self.mamba       = Mamba2(d_model=hidden_size, d_state=d_state, d_conv=4, expand=expand)
        else:
            self.mamba       = Mamba(d_model=hidden_size, d_state=d_state, d_conv=4, expand=expand) 
        self.norm2       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim   = int(hidden_size * mlp_ratio)
        approx_gelu      = lambda: nn.GELU(approximate="tanh")
        self.mlp         = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,out_features=hidden_size, act_layer=approx_gelu, drop=0)
        self.zz_paths     = zz_paths
        self.zz_paths_rev = zz_paths_rev
        self.layer_id     = layer_id

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        tmp  = x
        path = self.zz_paths[self.layer_id % len(self.zz_paths)]
        path_rev = self.zz_paths_rev[self.layer_id % len(self.zz_paths_rev)]
        tmp1 = tmp[:, path, :]        # zigzag 順に並び替え
        tmp1 = self.mamba(self.norm1(tmp1))       # Mamba処理
        tmp2 = tmp1[:, path_rev, :]   # 元の順序に戻す
        x = x + tmp2
        x = x + self.mlp(self.norm2(x)) 
        return x
 
class MambaDownBlock(nn.Module):
    def __init__(self, input_size, patch_size, hidden_dim, num_blocks=2, scan_type="zigzagN8", **kwargs):
        super().__init__()
        self.downsample = PatchMerging((input_size, input_size), hidden_dim)
        zz_paths, zz_paths_rev = create_path(input_size, patch_size, scan_type=scan_type)
        
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim*2, **kwargs,zz_paths=zz_paths, zz_paths_rev=zz_paths_rev, layer_id=i) for i in range(num_blocks)
        ])

    def forward(self, x, skip=None):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x, skip)
        return x
    
class MambaUpBlock(nn.Module):
    def __init__(self, input_size, patch_size, hidden_dim, num_blocks=2, scan_type="zigzagN8", **kwargs):
        super().__init__()
        zz_paths, zz_paths_rev = create_path(input_size*2, patch_size, scan_type=scan_type)
        self.upsample = PatchExpand((input_size//2, input_size//2), hidden_dim*2)
        self.conv1d   = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=1)  # skip connection で結合した後の次元を調整するための1x1畳み込み
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim, **kwargs, zz_paths=zz_paths, zz_paths_rev=zz_paths_rev, layer_id=i) for i in range(num_blocks)
        ])

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=-1)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # (B, hidden_dim*2, T) -> (B, hidden_dim, T)
        for block in self.blocks:
            x = block(x)
        return x

class MambaEncoder(nn.Module):
    def __init__(self, input_size, patch_size, hidden_size, num_heads, mlp_ratio=4.0, scan_type="zigzagN8", **kwargs):
        super().__init__()
        
        self.zz_paths, self.zz_paths_rev = create_path(input_size, patch_size, scan_type)
        self.enc1 = nn.ModuleList([
            MambaBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, skip=False, zz_paths=self.zz_paths, zz_paths_rev=self.zz_paths_rev, layer_id=id) for id in range(2)
        ])
        self.down1 = MambaDownBlock(input_size//2, patch_size, hidden_size, num_blocks=2, **kwargs)    # input_size//2 => input_size//4
        self.down2 = MambaDownBlock(input_size//4, patch_size, hidden_size*2, num_blocks=2, **kwargs)  # input_size//4 ⇒ input_size//8
        self.down3 = MambaDownBlock(input_size//8, patch_size, hidden_size*4, num_blocks=2, **kwargs)  # input_size//8 ⇒ input_size//16
        
    def forward(self, x):
        for block in self.enc1:
            x = block(x)
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        return x, [x1, x2, x3]

class MambaDecoder(nn.Module):
    def __init__(self, input_size, patch_size, hidden_size, num_heads, mlp_ratio=4.0, scan_type="zigzagN8", **kwargs):
        super().__init__()
        self.zz_paths, self.zz_paths_rev = create_path(input_size, patch_size, scan_type)
        self.up3 = MambaUpBlock(input_size//8, patch_size, hidden_size*4, num_blocks=2, **kwargs)  # input_size//16 ⇒ input_size//8
        self.up2 = MambaUpBlock(input_size//4, patch_size, hidden_size*2, num_blocks=2, **kwargs)  # input_size//8 ⇒ input_size//4
        self.up1 = MambaUpBlock(input_size//2, patch_size, hidden_size, num_blocks=2, **kwargs)    # input_size//4 ⇒ input_size//2
        self.dec1 = nn.ModuleList([
            MambaBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, skip=False, zz_paths=self.zz_paths, zz_paths_rev=self.zz_paths_rev, layer_id=id) for id in range(2)
        ])
    
    def forward(self, x, skips):
        x = self.up3(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        for block in self.dec1:
            x = block(x)
        return x

class UMamba(nn.Module):
    def __init__(self, input_size, patch_size, hidden_size, num_heads, mlp_ratio=4.0, scan_type="zigzagN8",in_channels=2, out_channels=2, **kwargs):
        super().__init__()
        self.patch_emb = PatchEmbed(img_size=input_size, patch_size=2, in_chans=in_channels, embed_dim=64, bias=True)
        num_patches = self.patch_emb.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.out_channels = out_channels
        self.encoder = MambaEncoder(input_size, patch_size, hidden_size, num_heads, mlp_ratio, scan_type, **kwargs)
        self.zz_paths, self.zz_paths_rev = create_path(input_size//8, patch_size, scan_type)
        self.bottle = nn.ModuleList([
            MambaBlock(hidden_size*8, num_heads=num_heads, mlp_ratio=mlp_ratio, skip=False, zz_paths=self.zz_paths, zz_paths_rev=self.zz_paths_rev, layer_id=id) for id in range(2)
        ])
        self.decoder = MambaDecoder(input_size, patch_size, hidden_size, num_heads, mlp_ratio, scan_type, **kwargs)
        self.final_layer = nn.Linear(hidden_size, patch_size**2 * out_channels, bias=True)
        self.final_conv  = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # パッチ化された出力を画像に変換するための1x1畳み込み
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_emb.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_emb.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_emb.proj.bias, 0)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_emb.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
        
    def forward(self, x):
        x = self.patch_emb(x) + self.pos_embed 
        x, skips = self.encoder(x)
        for block in self.bottle:
            x = block(x)
        x = self.decoder(x, skips)
        x = self.final_layer(x)
        
        x = self.unpatchify(x)             # (N, out_channels, H, W)
        x = self.final_conv(x)              # パッチ化された出力を画像に変換

        return x
    
    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
