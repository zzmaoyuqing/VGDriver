## from https://github.com/lucidrains/vit-pytorch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
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
    def __init__(self, dim, dropout, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dropout, heads = heads, dim_head = dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, model_config, pool='cls'):
        super().__init__()
        image_height, image_width = pair(model_config['image_size'])  # 224 * 224 * 3
        patch_height, patch_width = pair(model_config['patch_size'])  # 16 * 16  * 3

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        channels = model_config['ch']
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, model_config['dim'])
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, model_config['dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_config['dim']))
        self.dropout = nn.Dropout(model_config['emb_dropout'])

        self.transformer = Transformer(model_config['dim'], model_config['depth'],
                                       model_config['heads'], model_config['dim_head'],
                                       model_config['mlp_dim'], model_config['dropout'])

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_config['dim']),
            nn.Linear(model_config['dim'], model_config['num_classes'])
        )

    def forward(self, input):

        # print('input.shape', input.shape) # torch.Size([128, 3, 224, 224])

        x = self.to_patch_embedding(input)  # torch.Size([128, 196, 256])
        b, n, _ = x.shape  # b:128 n:196(patch_dim) -:256(dim)

        # add class token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # torch.Size([128, 1, 256])

        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([128, 197, 256])
        # positional embedding
        x += self.pos_embedding[:, :(n + 1)]   # torch.Size([128, 197, 256])
        x = self.dropout(x)

        x = self.transformer(x)                # torch.Size([128, 197, 256])

        # extract class token
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # torch.Size([128, 256])

        x = self.to_latent(x)    # torch.Size([128, 256])
        return self.mlp_head(x)  # torch.Size([32, 1])


