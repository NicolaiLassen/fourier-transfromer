import torch.nn as nn
import torch

from functools import wraps
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

# helpers

def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, pos, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.pos = pos

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        q = self.pos.rotate_queries_or_keys(q)
        k = self.pos.rotate_queries_or_keys(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        
        # https://arxiv.org/abs/2104.09864
        pos = RotaryEmbedding(dim = dim_head)
        
        self.norm = nn.LayerNorm(dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(pos, dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Attention(pos, dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                nn.LayerNorm(dim)
            ]))
             
    def forward(self, x):
        b = x.shape[0]
       
        # https://arxiv.org/abs/2205.15868
        for attn_s, attn_t, norm in self.layers:
            
            x_s = rearrange(x, 'b t p d -> (b t) p d')
            x_s = rearrange(attn_s(x_s), '(b t) p d -> b t p d', b=b)
            
            x_t = rearrange(x + x_s, 'b t p d -> (b p) t d')    
            x_t = rearrange(attn_t(x_t), '(b p) t d -> b t p d', b=b)
            
            x = norm(x_s + x_t) + x
            
        return self.norm(x)

# https://arxiv.org/abs/2010.11929
class TVit(nn.Module):
    def __init__(self, *, dim, depth, heads, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = SpatioTemporalTransformer(dim, depth, heads, dim_head, dropout)
        self.mlp = nn.Linear(dim, dim)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp(x)

# https://arxiv.org/abs/2010.11929
class STAViT(nn.Module):
    def __init__(self,
                 channels=1, 
                 patch_size=64,
                 dim=256, 
                 heads=4,
                 depth=2,
                 dim_head=64
                 ):
        super().__init__()
        
        pixel_values_per_patch = channels * patch_size * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(pixel_values_per_patch, dim),
        )
        
        self.spatio_temporal_transformer = TVit(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head=dim_head,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.to_pixels = nn.Linear(dim, pixel_values_per_patch, bias = False)
            
    def to_patches(self, x):
        return self.to_patch_embedding(x)

    def embed(self, x):
        return self.spatio_temporal_transformer(x)
        
    def recover(self, x, h, w):
        p = self.to_pixels(x)
        p = rearrange(p, 'b t p d -> b t (p d)')
        p = rearrange(p, 'b t (d h w) -> b t d h w', h=h, w=w)
        return p
        
    def forward(self, x):        
        h, w = x.shape[3:5]
        x = self.to_patches(x)
        x_h = self.embed(x)
        
        x_h_t_p = self.recover(x_h,h,w)
        x_p = self.recover(x,h,w)
        
        return x, x_h, x_p, x_h_t_p