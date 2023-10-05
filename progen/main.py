import math

import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from torch import einsum, nn

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)


#constants
ATTN_MASK_VALUE = -1e10


# helpers

class LayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        create_scale: bool = True,
        create_offset: bool = False    
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            dim,
            elementwise_affine=create_scale
        )
        self.offset = nn.Parameter(
            torch.zeros(dim)) if create_offset else None
    
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
    sinusoid_inp = repeat(
        sinusoid_inp,
        "b n -> b (n r)",
        r=2
    )[None, :, :]
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=-1)


def rotate_every_two(x):
    """
    x = torch.randn(1, 4, 4)
    x = rotate_every_two(x)
    print(x)
    
    """
    x = rearrange(
        x,
        "... (d r) -> ... d r",
        r = 2
    )
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack((
        -x2,
        x1
    ), axis=-1)

    return rearrange(
        x,
        "... d r -> ... (d r)"
    )


# def apply_rotary_pos_emb(x, sincos):
#     """
#     Rotary position embedding

#     sincos = fixed_pos_embeddings(4, 4)
#     print(sincos)
#     x = torch.randn(1, 4, 4)
#     x = apply_rotary_pos_emb(x, sincos)
#     """
#     sin, cos = sincos
#     rot_dim = sin.shape[-1]
#     x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

#     x = (x * cos) + (rotate_every_two(x) * sin)
#     return torch.cat((x, x_pass), axis=-1)

# shift tokens by 1 to the right, and wrap the last token to the first position

def shift_tokens(x):
    """
    Shift tokens

    x = torch.randn(1, 4, 4)
    x = shift_tokens(x)
    print(x)
    """
    x = torch.cat((
        x[..., -1:, :],
        x[..., :-1, :]
    ), dim=-2)
    return x

# classes
class SGU(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        seq_len,
        eps=1e-3
    ):
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
        n = self.seq_len
        x, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)

        weights = self.spatial_weights.tril()
        gate = torch.einsum("n d, m n -> m d", gate, weights)
        gate += self.spatial_biases

        x = x * gate
        return self.proj_out(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        glu: bool = False,
        seq_len: int = None,
        spatial_gate: bool = False,
        shift_tokens: bool = True
    ):
        super().__init__()

        #checks
        assert isinstance(dim, int), "Expected 'dim' to be an integer"
        assert isinstance(ff_mult, int), "Expected 'ff_mult' to be an integer"
        assert isinstance(glu, bool), "Expected 'glu' to be a boolean"
        assert isinstance(seq_len, int), "Expected 'seq_len' to be an integer"
        assert isinstance(spatial_gate, bool), "Expected 'spatial_gate' to be a boolean"
        assert isinstance(shift_tokens, bool), "Expected 'shift_tokens' to be a boolean"

        hidden_dim = dim * ff_mult
        self.glu = glu
        
        if glu:
            hidden_dim *= 2
        
        self.norm = nn.LayerNorm(dim)
        self.shift_tokens = shift_tokens

        self.proj_in = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

        self.sgu = SGU(
            dim=hidden_dim,
            dim_out= hidden_dim // 2,
            seq_len=seq_len
        ) if spatial_gate else None

    def forward(self, x):
        x = self.norm(x)

        if self.shift_tokens:
            x = shift_tokens(x)
        
        x = self.proj_in(x)

        if self.glu:
            x, gate = x.chunk(2, dim=-1)
            x *= torch.sigmoid(gate)
        else:
            x = F.gelu(x)
        
        if self.sgu:
            x = self.sgu(x)

        return self.proj_out(x)
    

class LocalAttention(nn.Module):
    """
        x = torch.randn(1, 4, 4)
        model = LocalAttention(window_size=2, causal=False, look_backward=1, look_forward=0, dropout=0., shared_qk=False, rel_pos_emb_config=None, dim=None, autopad=False, exact_windowsize=False, scale=None, use_rotary_pos_emb=True, use_xpos=False, xpos_scale_base=None)
        out  = model(x, x, x)
        print(out)
    """
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False,
        scale = None,
        use_rotary_pos_emb = True,
        use_xpos = False,
        xpos_scale_base = None
    ):
        super().__init__()
        
        #assertion type checks
        # assert isinstance(window_size, int), "Expected 'window_size' to be an integer"
        # assert isinstance(causal, bool), "Expected 'causal' to be a boolean"
        # assert isinstance(look_backward, int), "Expected 'look_backward' to be an integer"
        # assert isinstance(look_forward, int), "Expected 'look_forward' to be an integer"
        # assert isinstance(dropout, float), "Expected 'dropout' to be a float"
        # assert isinstance(shared_qk, bool), "Expected 'shared_qk' to be a boolean"
        # assert isinstance(rel_pos_emb_config, tuple), "Expected 'rel_pos_emb_config' to be a tuple"
        # assert isinstance(dim, int), "Expected 'dim' to be an integer"
        # assert isinstance(autopad, bool), "Expected 'autopad' to be a boolean"
        # assert isinstance(exact_windowsize, bool), "Expected 'exact_windowsize' to be a boolean"
        # assert isinstance(scale, float), "Expected 'scale' to be a float"
        # assert isinstance(use_rotary_pos_emb, bool), "Expected 'use_rotary_pos_emb' to be a boolean"
        # assert isinstance(use_xpos, bool), "Expected 'use_xpos' to be a boolean"
        # assert isinstance(xpos_scale_base, int), "Expected 'xpos_scale_base' to be an integer"

        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos
        
        # backwards compatible with old `rel_pos_emb_config` deprecated argument
        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)):  
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos = use_xpos,
                scale_base = default(xpos_scale_base, window_size // 2)
            )

    def forward(
        self,
        q, k, v,
        mask = None,
        input_mask = None,
        attn_bias = None,
        window_size = None
    ):

        mask = default(mask, input_mask)

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(
                lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v)
            )

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing

        bq, bk, bv = map(
            lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v)
        )

        bq = bq * scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = (self.window_size * self.look_backward)
            max_forward_window_size = (self.window_size * self.look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out


class ProGenBase(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        seq_len,
        depth,
        window_size = 512,
        global_mlp_depth = 2,
        heads = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        ff_glu: bool = True,
        attn_dim = None,
        clamp_gate = True,
        shift_tokens = True,
        dropout = 0.,
    ):
        super(ProGenBase, self).__init__()
        self.dim_head = dim_head
        self.embed = nn.Embedding(num_tokens, dim)
        self.dropout = dropout

        self.attn_layers = nn.ModuleList([])
        self.ff_layers = nn.ModuleList([])

        for i in range(depth):
            use_gmlp = (depth - i) <= global_mlp_depth
            use_ff_glu = use_gmlp and ff_glu
            
            #add attention to module list
            self.attn_layers.append(
                LocalAttention(
                    dim=self.dim_head,
                    window_size=window_size,
                    causal=True,
                    look_backward=1,
                    look_forward=0,
                    dropout=self.dropout,
                    exact_windowsize=True,
                )
            )
            
            #add feedforward to module list
            self.ff_layers.append(
                FeedForward(
                    FeedForward(
                        dim=dim,
                        ff_mult=ff_mult,
                        seq_len=seq_len,
                        spatial_gate=use_gmlp,
                        glu=use_ff_glu,
                        shift_tokens=shift_tokens,
                    )
                )
            )
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    
    def forward(self, x):
        #embed tokens
        x = self.embed(x)

        #add mask
        mask = torch.ones(
            x.size(0),
            x.size(1),
        ).bool().to(x.device)

        # apply attn, ff
        for attn, ff in zip(self.attn_layer, self.ff_layers):
            q = k = v = x
            x = x + attn(q, k, v, mask=mask)
            x = x + ff(x)
        
        return self.to_logits(x)

model = ProGenBase(dim=512, num_tokens=20000, seq_len=2048, depth=6)
x = torch.randint(0, 20000, (1, 2048))
out = model(x)
print(out.shape)
