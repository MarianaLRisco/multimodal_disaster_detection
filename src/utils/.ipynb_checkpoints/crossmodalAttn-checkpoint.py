import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):

        super().__init__()

        self.Wt = nn.Linear(embedding_dim, hidden_dim)
        self.Wi = nn.Linear(embedding_dim, hidden_dim)

        self.Es = nn.Linear(embedding_dim * 2, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.tanh = nn.Tanh()

    def forward(self, h_t, h_i):

        ht_bar = self.tanh(self.Wt(h_t))
        hi_bar = self.tanh(self.Wi(h_i))

        ht_i = self.tanh(
            self.Es(torch.cat([h_t, h_i], dim=1))
        )

        hi_t = self.tanh(
            self.Es(torch.cat([h_i, h_t], dim=1))
        )

        Xm = torch.stack(
            [ht_bar, hi_bar, ht_i, hi_t],
            dim=1
        )  # [B,4,D]

        attn_out, _ = self.cross_attn(
            Xm,
            Xm,
            Xm
        )

        attn_out = self.norm(attn_out + Xm)

        fused = attn_out.reshape(
            attn_out.size(0),
            -1
        )

        return fused