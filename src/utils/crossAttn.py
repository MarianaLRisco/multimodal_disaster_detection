import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):

    def __init__(
        self,
        text_dim,
        image_dim,
        proj_dim=512
    ):
        super().__init__()

        # projections

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # cross-attention masks

        self.text_gate = nn.Linear(
            image_dim,
            proj_dim
        )

        self.image_gate = nn.Linear(
            text_dim,
            proj_dim
        )

        # fusion

        # self.fusion = nn.Sequential(

        #     nn.Linear(
        #         proj_dim * 2,
        #         proj_dim * 2
        #     ),

        #     nn.BatchNorm1d(
        #         proj_dim * 2
        #     ),

        #     nn.ReLU(),

        #     nn.Dropout(0.3)
        # )

    def forward(
        self,
        text_emb,
        image_emb
    ):

        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)

        alpha_text = torch.sigmoid(
            self.text_gate(image_emb)
        )

        alpha_image = torch.sigmoid(
            self.image_gate(text_emb)
        )

        text_masked = alpha_text * text_proj
        image_masked = alpha_image * image_proj

        joint = torch.cat(
            [
                text_masked,
                image_masked
            ],
            dim=-1
        )

        return joint