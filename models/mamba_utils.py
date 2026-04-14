from einops import rearrange
import torch
import numpy as np
import math
from timm.models.vision_transformer import Attention
from torch import nn


def zigzag_path(N):
    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else N - 1 - j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths

def reverse_permut_np(permutation):
    n = len(permutation)
    reverse = np.array([0] * n)
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse



import numpy as np
import numpy as np

# 方向数に応じて使うコーナーが決まる
# 1方向: TL のみ
# 2方向: TL, BR
# 3方向: TL, BR, TR
# 4方向: TL, BR, TR, BL

def raster_path(N, num_directions=4):
    """
    ラスタースキャンのパスを生成する。zigzag_path と同じインターフェース。
    num_directions: 1〜4
      1: TL のみ (左上→右下)
      2: TL, BR
      3: TL, BR, TR
      4: TL, BR, TR, BL

    Returns:
        list of np.array, shape (N*N,)
    """
    assert 1 <= num_directions <= 4

    CONFIGS = [
        (1,  1),   # TL: 行↓, 列→
        (-1, -1),  # BR: 行↑, 列←
        (1,  -1),  # TR: 行↓, 列←
        (-1,  1),  # BL: 行↑, 列→
    ]

    paths = []
    for dir_row, dir_col in CONFIGS[:num_directions]:
        row_range = range(N) if dir_row == 1 else range(N - 1, -1, -1)
        col_range = range(N) if dir_col == 1 else range(N - 1, -1, -1)
        path = []
        for i in row_range:
            for j in col_range:
                path.append(i * N + j)
        paths.append(np.array(path))

    return paths
# N = 4
# paths = raster_path(N, num_directions=2)

# for i, p in enumerate(paths):
#     rev = reverse_permut_np(p)
#     # p[rev[k]] == k が全kで成立するはず
#     assert all(p[rev[k]] == k for k in range(N * N)), "逆変換ミス！"
#     print(f"path {i}: OK")
#     print(f"  path    : {p}")
#     print(f"  reverse : {rev}")
# from mamba_ssm import Mamba
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = torch.randn(1,4,32,32).to(device)

# class ZigMamba(nn.Module):
#     def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, scan_type="zigzagN8",img_dim=32, patch_size=2):
#         super().__init__()
#         self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
#         self.scan_type=scan_type
#         self.zigzag_num = int(scan_type.replace("zigzagN", ""))
#         self.num_patches = (img_dim // patch_size) ** 2
#         patch_side_len = int(math.sqrt(self.num_patches))
        
#         if scan_type.startswith("zigzagN") or scan_type.startswith("parallelN"):
#             _zz_paths = zigzag_path(N=patch_side_len)
#             if scan_type.startswith("zigzagN"):
#                 zigzag_num = int(scan_type.replace("zigzagN", ""))
#                 zz_paths = _zz_paths[:zigzag_num]
#                 assert (
#                     len(zz_paths) == zigzag_num
#                     ), f"{len(zz_paths)} != {zigzag_num}"
#             elif scan_type.startswith("parallelN"):
#                 zz_paths  = _zz_paths[:8]
#         self.zz_paths     = zz_paths
#         self.zz_paths_rev = [reverse_permut_np(_) for _ in zz_paths]
        
#     def forward(self, x_seq):
#         return self.mamba(x_seq)


# # scan_type="zigzagN8"
# # zigzag_num = int(scan_type.replace("zigzagN", ""))
# # img_dim = 32
# # patch_size = 8
# # num_patches = (img_dim // patch_size) ** 2
# # patch_side_len = int(math.sqrt(num_patches))

# # if scan_type.startswith("zigzagN") or scan_type.startswith("parallelN"):
# #     _zz_paths = zigzag_path(N=patch_side_len)
# #     if scan_type.startswith("zigzagN"):
# #         zigzag_num = int(scan_type.replace("zigzagN", ""))
# #         zz_paths = _zz_paths[:zigzag_num]
# #         assert (
# #             len(zz_paths) == zigzag_num
# #             ), f"{len(zz_paths)} != {zigzag_num}"
# #     elif scan_type.startswith("parallelN"):
# #         zz_paths = _zz_paths[:8]
        
# # zz_paths_rev = [reverse_permut_np(_) for _ in zz_paths]

# # torch テンソルに変換（depth 分リピートは不要 ─ 1ブロックなので）
# # zz_paths_t     = [torch.from_numpy(p).long().to(device) for p in zz_paths]
# # zz_paths_rev_t = [torch.from_numpy(p).long().to(device) for p in zz_paths_rev]


# # ── Step1: パッチ埋め込み ──────────────────────
# print(x.shape)
# patch_embed = nn.Conv2d(4, 64, kernel_size=patch_size, stride=patch_size).to(device)
# x_seq = rearrange(patch_embed(x), "b d h w -> b (h w) d")
# print(f"Patch Embedded Shape: {x_seq.shape}")  # (1, 256, 64)

# # ── Step3: Mamba ──────────────────────────────
# mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
# attn  = Attention(64, num_heads=8).to(device)
# # ── Step4: 8方向スキャン ──────────────────────
# out = torch.zeros_like(x_seq)

# for path, path_rev in zip(zz_paths_t, zz_paths_rev_t):
#     x_perm   = x_seq[:, path, :]        # zigzag 順に並び替え
#     print(f"Permuted Shape: {x_perm.shape}")  # (1, 256, 10)
#     y        = mamba(x_perm)            # Mamba forward
#     out     += y[:, path_rev, :]        # 元の空間順に戻して加算
    

# # メモリ使用量を比較(Mamba vs Attention)

