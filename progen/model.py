import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from progen.local_attention import LocalAttention

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions


def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def to(t):
    return {"device": t.device, "dtype": t.dtype}


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim=-1)
    return normed.type(dtype)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


# constants
ATTN_MASK_VALUE = -1e10


# helpers


class LayerNorm(nn.Module):
    def __init__(self, dim, create_scale: bool = True, create_offset: bool = False):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=create_scale)
        self.offset = nn.Parameter(torch.zeros(dim)) if create_offset else None

    def forward(self, x):
        x = self.norm(x)
        if self.offset is not None:
            x += self.offset
        return x


def fixed_pos_embeddings(seq, dim):
    """
    Fixed pos embedding


    x = fixed_pos_embeddings(10, 512)
    print(x)

    """
    inv_freq = 1.0 / (1000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j",
        torch.arange(seq),
        inv_freq,
    )
    sinusoid_inp = repeat(sinusoid_inp, "b n -> b (n r)", r=2)[None, :, :]
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=-1)


def rotate_every_two(x):
    """
    x = torch.randn(1, 4, 4)
    x = rotate_every_two(x)
    print(x)

    """
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack((-x2, x1), axis=-1)

    return rearrange(x, "... d r -> ... (d r)")


def shift_tokens(x):
    """
    Shift tokens

    x = torch.randn(1, 4, 4)
    x = shift_tokens(x)
    print(x)
    """
    x = torch.cat((x[..., -1:, :], x[..., :-1, :]), dim=-2)
    return x


# classes
class SGU(nn.Module):
    def __init__(self, dim, dim_out, seq_len, eps=1e-3):
        super().__init__()
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim_out)

        init_scale = eps / seq_len
        self.spatial_weights = nn.Parameter(
            torch.empty(seq_len, seq_len).uniform_(-init_scale, init_scale)
        )
        self.spatial_biases = nn.Parameter(torch.ones(seq_len, 1))

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)

        weights = self.spatial_weights.tril()
        gate = torch.einsum("n d, m n -> m d", gate, weights)
        gate += self.spatial_biases

        x = x * gate
        return self.proj_out(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ProGen(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        seq_len,
        depth,
        window_size=512,
        global_mlp_depth=2,
        heads=8,
        dim_head: int = 512,
        ff_mult: int = 4,
        ff_glu: bool = True,
        attn_dim=None,
        clamp_gate=True,
        shift_tokens=True,
        dropout=0.0,
    ):
        super(ProGen, self).__init__()
        self.dim_head = dim_head
        self.embed = nn.Embedding(num_tokens, dim)
        self.dropout = dropout

        self.attn_layers = nn.ModuleList([])
        self.ff_layers = nn.ModuleList([])

        for i in range(depth):
            # add attention to module list
            self.attn_layers.append(
                LocalAttention(
                    dim=dim_head,
                    window_size=window_size,
                    causal=True,
                    look_backward=1,
                    look_forward=0,
                    dropout=self.dropout,
                    exact_windowsize=True,
                )
            )

            # add feedforward to module list
            self.ff_layers.append(
                FeedForward(
                    dim=dim,
                    mult=ff_mult,
                    dropout=self.dropout,
                )
            )

        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        # embed tokens
        x = self.embed(x)

        # add mask
        mask = (
            torch.ones(
                x.size(0),
                x.size(1),
            )
            .bool()
            .to(x.device)
        )

        # apply attn, ff
        for attn, ff in zip(self.attn_layers, self.ff_layers):
            q = k = v = x
            x = x + attn(q, k, v, mask=mask)
            x = x + ff(x)

        return self.to_logits(x)
