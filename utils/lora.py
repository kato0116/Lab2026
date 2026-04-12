import torch
import torch.nn as nn

class LoRAConv2d(nn.Module):
    def __init__(self, base_conv, rank=4, alpha=1.0):
        super().__init__()
        self.base = base_conv
        in_c, out_c = base_conv.in_channels, base_conv.out_channels
        self.down = nn.Conv2d(in_c, rank, kernel_size=1, bias=False)
        self.up   = nn.Conv2d(rank, out_c, kernel_size=1, bias=False)
        self.scale = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank=4, alpha=1.0):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        mid_features = self.in_features // rank
        self.scale = alpha / rank
        # LoRA学習パラメータ
        self.down = nn.Linear(self.in_features, rank, bias=False)
        self.up   = nn.Linear(rank, self.out_features, bias=False)
        # 元のLinearのパラメータは固定（freeze）
        for p in self.base.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def insert_lora_adapters(vae, rank=4, alpha=1.0):
    """
    AutoencoderKL に LoRAConv2d / LoRALinear アダプタを挿入する

    Args:
        vae: AutoencoderKL モデル
        rank (int): LoRAのランク
        alpha (float): スケーリング係数（alpha / rank が掛かる）

    Returns:
        LoRAを差し込んだAutoencoderKL
    """

    # --- Encoder の ResNet Block ---
    for i in [2, 3]:
        for j, block in enumerate(vae.encoder.down_blocks[i].resnets):
            conv = block.conv1
            block.conv1 = LoRAConv2d(conv, rank=rank, alpha=alpha)

    # --- Decoder の ResNet Block ---
    for i in [2, 3]:
        for j, block in enumerate(vae.decoder.up_blocks[i].resnets):
            conv = block.conv1
            block.conv1 = LoRAConv2d(conv, rank=rank, alpha=alpha)

    # --- Attention (Encoder Mid Block) ---
    attn = vae.encoder.mid_block.attentions[0]
    attn.to_q = LoRALinear(attn.to_q, rank=rank, alpha=alpha)
    attn.to_k = LoRALinear(attn.to_k, rank=rank, alpha=alpha)
    attn.to_v = LoRALinear(attn.to_v, rank=rank, alpha=alpha)
    attn.to_out[0] = LoRALinear(attn.to_out[0], rank=rank, alpha=alpha)

    # --- Attention (Decoder Mid Block) ---
    attn = vae.decoder.mid_block.attentions[0]
    attn.to_q = LoRALinear(attn.to_q, rank=rank, alpha=alpha)
    attn.to_k = LoRALinear(attn.to_k, rank=rank, alpha=alpha)
    attn.to_v = LoRALinear(attn.to_v, rank=rank, alpha=alpha)
    attn.to_out[0] = LoRALinear(attn.to_out[0], rank=rank, alpha=alpha)

    return vae
# from diffusers import AutoencoderKL
# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
# # パラメータをフリーズ
# for param in vae.parameters():
#     param.requires_grad = False
# vae = insert_lora_adapters(vae, rank=4, alpha=1.0)

# # requires_grad=True なパラメータ一覧を出力
# print("\n=== Trainable Parameters ===")
# for name, param in vae.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)