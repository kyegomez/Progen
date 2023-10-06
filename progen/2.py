import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# constants
ATTN_MASK_VALUE = -1e10

# helpers
def fixed_pos_embedding(seq, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq), inv_freq)
    
    # Repeat the sinusoid_inp along the last dimension
    sinusoid_inp = sinusoid_inp.unsqueeze(-1).repeat(1, 1, 2)  # Instead of rearrange
    
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    rot_dim = sin.shape[-1]
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x = (x * cos) + (rotate_every_two(x) * sin)
    return torch.cat((x, x_pass), dim=-1)

def shift_tokens(x):
    x_shift, x_pass = torch.chunk(x, 2, dim=-1)
    x_shift = F.pad(x_shift, (0, 0, 1, 0), "constant", 0)[:-1]
    return torch.cat((x_shift, x_pass), dim=-1)

# classes
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        return self.layer_norm(x)

# Continuing with the classes

class LocalAttention(nn.Module):
    def __init__(self, dim, window_size, heads=8, dim_head=64, shift_tokens=True):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.shift_tokens = shift_tokens

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, pos_emb):
        x = self.norm(x)

        if self.shift_tokens:
            x = shift_tokens(x)

        n, _, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        q, k, v = map(lambda t: t.view(self.heads, *t.shape[:-1]), (q, k, v))
        q, k, v = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k, v))

        window = n // self.window_size

        # In LocalAttention's forward method
        q, k, v = map(lambda t: rearrange(t, 'h (w n) d -> h w n d', w=window), (q, k, v))
        k, v = map(lambda t: F.pad(t, (0, 0, 0, 1), "constant", 0), (k, v))  # Adjust padding for clarity
        k, v = map(lambda t: torch.cat((t[:, :-1], t[:, 1:]), dim=2), (k, v))

        sim = torch.einsum('h w i d, h w j d -> h w i j', q, k) * self.scale
        mask = torch.tril(torch.ones(self.window_size, self.window_size * 2)).to(x.device)
        sim.masked_fill_(~mask.bool(), ATTN_MASK_VALUE)
        attn = F.softmax(sim, dim=-1)
        
        out = torch.einsum('h w i j, h w j d -> h w i d', attn, v)
        out = rearrange(out, 'h w n d -> (w n) (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4, glu=False, seq_len=None, spatial_gate=False, shift_tokens=True):
        super().__init__()
        hidden_dim = dim * ff_mult
        hidden_dim *= (2 if glu else 1)

        self.norm = LayerNorm(dim)
        self.shift_tokens = shift_tokens

        self.proj_in = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

        self.glu = glu
        self.sgu = SGU(dim=hidden_dim, dim_out=hidden_dim // 2, seq_len=seq_len) if spatial_gate else None

    def forward(self, x):
        x = self.norm(x)

        if self.shift_tokens:
            x = shift_tokens(x)

        x = self.proj_in(x)

        if self.glu:
            x, gate = torch.chunk(x, 2, dim=-1)
            x *= torch.sigmoid(gate)
        else:
            x = F.gelu(x)

        if self.sgu is not None:
            x = self.sgu(x)

        x = self.proj_out(x)
        return x

class SGU(nn.Module):
    def __init__(self, dim, dim_out, seq_len, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.seq_len = seq_len
        self.norm = LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim_out)

        init_scale = self.eps / seq_len
        self.weights = nn.Parameter(torch.empty(seq_len, seq_len).uniform_(-init_scale, init_scale))
        self.biases = nn.Parameter(torch.ones(seq_len, 1))

    def forward(self, x):
        x, gate = torch.chunk(x, 2, dim=-1)

        gate = self.norm(gate)

        weights = self.weights.tril(diagonal=0)  # explicitly mention diagonal for clarity
        gate = torch.einsum('n d, m n -> m d', gate, weights)
        gate += self.biases

        x *= gate
        return self.proj_out(x)

class ProGenBase(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, window_size=256, global_mlp_depth=2, heads=8, dim_head=64, ff_mult=4, ff_glu=True, attn_dim=None, clamp_gate=True, shift_tokens=True):
        super().__init__()
        self.dim_head = dim_head
        self.embed = nn.Embedding(num_tokens, dim)
        
        layers = []
        for i in range(depth):
            use_gmlp = (depth - i) <= global_mlp_depth
            use_ff_glu = not use_gmlp and ff_glu

            layers.extend([
                LocalAttention(dim=dim, window_size=window_size, heads=heads, dim_head=dim_head, shift_tokens=shift_tokens),
                FeedForward(dim=dim, ff_mult=ff_mult, glu=use_ff_glu, seq_len=seq_len, spatial_gate=use_gmlp, shift_tokens=shift_tokens)
            ])

        self.layers = nn.ModuleList(layers)
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        n = x.shape[0]
        x = self.embed(x)
        rotary_emb = fixed_pos_embedding(n, self.dim_head)

        for layer in self.layers:
            if isinstance(layer, LocalAttention):
                x = x + layer(x, pos_emb=rotary_emb)
            else:
                x = x + layer(x)

        return self.to_logits(x)


import torch

# Define some constants
NUM_TOKENS = 1000  # for instance, a vocabulary size
DIM = 512  # dimension of embeddings and model
SEQ_LEN = 128  # length of sequences
DEPTH = 4  # depth of network
WINDOW_SIZE = 32  # size of attention window
GLOBAL_MLP_DEPTH = 2  # depth of global MLP layers
HEADS = 8  # number of attention heads
DIM_HEAD = 64  # dimension of individual attention head
FF_MULT = 4  # feed-forward dimension multiplier
FF_GLU = True  # use GLU in feed-forward
SHIFT_TOKENS = True  # shift tokens in local attention and feed-forward

# Instantiate the model
model = ProGenBase(
    num_tokens=NUM_TOKENS,
    dim=DIM,
    seq_len=SEQ_LEN,
    depth=DEPTH,
    window_size=WINDOW_SIZE,
    global_mlp_depth=GLOBAL_MLP_DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD,
    ff_mult=FF_MULT,
    ff_glu=FF_GLU,
    shift_tokens=SHIFT_TOKENS
)

# Generate a random sequence of tokens
batch_size = 16
random_sequences = torch.randint(0, NUM_TOKENS, (batch_size, SEQ_LEN))

# Forward the sequences through the model
output_logits = model(random_sequences)

print(output_logits.shape)  # This should print [batch_size, SEQ_LEN, NUM_TOKENS]
