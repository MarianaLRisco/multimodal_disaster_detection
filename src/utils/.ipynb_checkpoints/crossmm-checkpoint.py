import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):

    def __init__(self, text_dim, image_dim, hidden_dim, num_heads=4):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        self.img_to_txt = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) #need_weights=True, average_attn_weights=False
        self.txt_to_img = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # MÁS CONTEXTO
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, text_tokens, image_tokens):

        t = self.text_proj(text_tokens)
        i = self.image_proj(image_tokens)

        i2t, _ = self.img_to_txt(query=i, key=t, value=t)
        t2i, _ = self.txt_to_img(query=t, key=i, value=i)

        i = i + i2t
        t = t + t2i

        # global pooling separado
        t_pool = t.mean(dim=1)
        i_pool = i.mean(dim=1)

        fused_cat = torch.cat([t_pool, i_pool, t_pool * i_pool, torch.abs(t_pool - i_pool)], dim=-1)

        gate = self.gate(fused_cat)

        fused = gate * t_pool + (1 - gate) * i_pool

        fused = self.norm(fused)
        fused = self.ffn(fused)

        return fused