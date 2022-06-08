import torch.nn as nn
import torch

from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

# helpers

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    
    def __init__(self, pos, in_dim):
        super(CrissCrossAttention,self).__init__()
        
        self.pos = pos
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=3)
        
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        
        proj_query = self.query_conv(x)
        proj_query = self.pos.rotate_queries_or_keys(proj_query)
        
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)
        proj_key = self.pos.rotate_queries_or_keys(proj_key)
        
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)

        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        return self.gamma*(out_H + out_W) + x

# classes

class TAViT(nn.Module):
    def __init__(self,
                 max_seq_len=15,
                 channels=1, 
                 patch_size=64,
                 dim=128,
                 recurrence=3,
                 dim_head=64,
                 dropout=0.01,
                 emb_dropout=0.1,
                 ):
        super().__init__()
        
        self.max_seq_len = max_seq_len        
        pixel_values_per_patch = channels * patch_size * patch_size
        
        self.to_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(pixel_values_per_patch, dim),
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        pos = RotaryEmbedding(dim = dim_head)
        
        modules = []
        for _ in range(recurrence):
                modules.append(CrissCrossAttention(pos, max_seq_len))
                modules.append(nn.Dropout(dropout))
    
        self.attn_layers = nn.Sequential(*modules)
        
        self.mlp = nn.Linear(dim, dim, bias=False)
        
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, pixel_values_per_patch)
        )
     
    def generate(self, start_tokens, seq_len):
        _, t, _, h, w = start_tokens.shape
                
        out = self.to_embedding(start_tokens)
        
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            
            last = self.next_latent(x)[:, -1:]
            out = torch.cat((out, last), dim = 1)
        
        out = out[:, t:]
        
        return model.recover(out, h, w)
        
    def next_latent(self, x_H0):
        x_H1 = self.attn_layers(x_H0)
        return self.mlp(x_H1)
    
    def recover(self, x, h, w):
        p = self.to_pixels(x)
        p = rearrange(p, 'b t p d -> b t (p d)')
        p = rearrange(p, 'b t (d h w) -> b t d h w', h=h, w=w)
        return p
        
    def forward(self, x):
        h, w = x.shape[3:5]
    
        x_e = self.to_embedding(x)
        x_h = self.next_latent(x_e)
        
        x_p = self.recover(x_e,h,w)
        x_h_p = self.recover(x_h,h,w)
        
        return x_e, x_h, x_p, x_h_p