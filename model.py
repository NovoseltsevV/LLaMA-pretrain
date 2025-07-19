import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
# import xformers.ops as xops
from torch.nn.attention import SDPBackend, sdpa_kernel
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float, Int
from typing import Tuple


@dataclass
class ModelConfig:
    d_model: int = 960
    n_layers: int = 15
    n_heads: int = 15
    d_head: int = 64
    d_ffn: int = 2560
    d_vocab: int = 32000
    context_len: int = 256
    rms_norm_eps: float = 1e-5
    new_context_len: int = 1024
    device: str = 'cuda'
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.ones(cfg.d_model))

    def forward(
        self,
        x: Float[Tensor, "batch L d_model"]
    ) -> Float[Tensor, "batch L d_model"]:
        # L -- length of each sentence in batch
        rms_norm = x.pow(2).mean(dim=-1, keepdim=True)
        norm = rms_norm + self.cfg.rms_norm_eps
        x_norm = x / torch.sqrt(norm)
        return x_norm * self.weight.view(1, 1, -1)


class RoPE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.interpolate = False
        assert self.cfg.d_head % 2 == 0, "RoPE requires even d_head"

    def set_interpolate(self, on: bool) -> None:
        self.interpolate = on

    def compute_rotation_matrices(
        self,
        seq_len: int
    ) -> Tuple[Float[Tensor, "1 1 seq_len half_d"],
               Float[Tensor, "1 1 seq_len half_d"]]:
        max_len = (self.cfg.new_context_len if self.interpolate
                   else self.cfg.context_len)
        if seq_len > max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum allowed length {max_len}"
            )
        half_d = self.cfg.d_head // 2
        indices = torch.arange(seq_len, device=self.cfg.device)
        if self.interpolate:
            indices = indices * self.cfg.context_len / self.cfg.new_context_len
        thetas = 10000.0 ** (-torch.arange(half_d, device=self.cfg.device)
                            / half_d)
        freqs = einsum(indices, thetas, "i, j -> i j")
        cos_matrix = torch.cos(freqs)[None, None, :, :]
        sin_matrix = torch.sin(freqs)[None, None, :, :]
        return cos_matrix, sin_matrix

    def rotate_matrix(
        self,
        x: Float[Tensor, "batch n_heads L d_head"]
    ) -> Float[Tensor, "batch n_heads L d_head"]:
        seq_len = x.shape[2]
        cos_matrix, sin_matrix = self.compute_rotation_matrices(seq_len)
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        out_even = x_even * cos_matrix - x_odd * sin_matrix
        out_odd = x_even * sin_matrix + x_odd * cos_matrix
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.rope = RoPE(cfg)
        self.w_q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.w_k = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.w_v = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.w_o = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))

        nn.init.normal_(self.w_q, std=cfg.init_std)
        nn.init.normal_(self.w_k, std=cfg.init_std)
        nn.init.normal_(self.w_v, std=cfg.init_std)
        nn.init.normal_(self.w_o, std=cfg.init_std)

    def forward(
        self,
        x: Float[Tensor, "batch L d_model"]
    ) -> Float[Tensor, "batch L d_model"]:
        q = einsum(self.w_q, x, "n dh dm, bs l dm -> bs l n dh")
        k = einsum(self.w_k, x, "n dh dm, bs l dm -> bs l n dh")
        v = einsum(self.w_v, x, "n dh dm, bs l dm -> bs l n dh")
        q_rot = self.rope.rotate_matrix(q.permute(0, 2, 1, 3))
        k_rot = self.rope.rotate_matrix(k.permute(0, 2, 1, 3))
        v = v.permute(0, 2, 1, 3)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        with torch.autocast(
            device_type=self.cfg.device, 
            dtype=torch.float16, 
            enabled=self.cfg.device == 'cuda'
        ):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                attn = F.scaled_dot_product_attention(
                    query=q_rot,
                    key=k_rot,
                    value=v,
                    dropout_p=0.1 if self.training else 0.0,
                    is_causal=True
                )

        output = einsum(self.w_o, attn, "n dm dh, bs n l dh -> bs l dm")
        return output


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.swish = nn.SiLU()
        self.layer1 = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.layer2 = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.layer3 = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)

        nn.init.normal_(self.layer1.weight, std=cfg.init_std)
        nn.init.normal_(self.layer2.weight, std=cfg.init_std)
        nn.init.normal_(self.layer3.weight, std=cfg.init_std)

    def forward(
        self,
        x: Float[Tensor, "batch L d_model"]
    ) -> Float[Tensor, "batch L d_model"]:
        x1 = self.swish(self.layer1(x))
        x2 = self.layer2(x)
        return self.layer3(x1 * x2)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: Float[Tensor, "batch L d_model"]
    ) -> Float[Tensor, "batch L d_model"]:
        attn_out = self.attn(self.ln1(x))
        resid = x + attn_out
        ffn_out = self.ffn(self.ln2(resid))
        out = resid + ffn_out
        return out


class MyLLaMA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(self.cfg.d_vocab, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)
        )
        self.unembed = nn.Linear(self.cfg.d_model, self.cfg.d_vocab)
        self.final_norm = RMSNorm(cfg)
        self.to(cfg.device)

    def interpolate(self, on: bool) -> None:
        for block in self.blocks:
            block.attn.rope.set_interpolate(on) # type: ignore

    def forward(
        self,
        x: Int[Tensor, "batch L"]
    ) -> Float[Tensor, "batch L d_vocab"]:
        output = self.embed(x)
        for block in self.blocks:
            output = block(output)
        output = self.final_norm(output)
        output = self.unembed(output)
        return output
