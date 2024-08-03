
import torch.nn as nn
import torch
import math
from einops import rearrange
class CARM(nn.Module):
    def __init__(self, channels):
        super(AltFilter, self).__init__()
        self.epi_trans = BasicTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer_1,buffer_2,I_u,I_v):
        buffer = torch.stack((buffer_1,buffer_2),dim = 2)
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()

        # Horizontal
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=I_u, v=I_v)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=I_u, v=I_v, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=I_u, v=I_v)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=I_u, v=I_v, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        buffer_1,buffer_2=torch.unbind(buffer,dim=2)
        return buffer_1,buffer_2


class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)
        

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   need_weights=False)[0] + epi_token
        
        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer