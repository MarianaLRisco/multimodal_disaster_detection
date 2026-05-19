import torch
import torch.nn as nn

from src.utils.crossmodalAttn import (
    CrossModalAttention
)


class CrossModalModel(nn.Module):

    def __init__(
        self,
        text_model,
        image_model,
        model_config,
        dataset_config
    ):

        super().__init__()

        self.text_model = text_model
        self.image_model = image_model

        self.model_config = model_config
        self.dataset_config = dataset_config

        # ORIGINAL DIMENSIONS

        self.text_dim = (
            text_model.embedding_dim
        )

        self.image_dim = (
            image_model.embedding_dim
        )

        # COMMON PROJECTION DIM

        self.projection_dim = (
            model_config[
                "projection_dim"
            ]
        )

        # PROJECTION LAYERS

        self.text_projection = nn.Linear(
            self.text_dim,
            self.projection_dim
        )

        self.image_projection = nn.Linear(
            self.image_dim,
            self.projection_dim
        )

        # CROSS MODAL ATTENTION

        self.cross_attention = (
            CrossModalAttention(

                embedding_dim=(
                    self.projection_dim
                ),

                hidden_dim=(
                    model_config[
                        "cross_hidden_dim"
                    ]
                )
            )
        )

        fusion_dim = (
            model_config[
                "cross_hidden_dim"
            ] * 4
        )

        # CLASSIFIER

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                model_config[
                    "classifier_hidden_dim"
                ]
            ),

            nn.ReLU(),

            nn.Dropout(
                model_config[
                    "dropout"
                ]
            ),

            nn.LayerNorm(
                model_config[
                    "classifier_hidden_dim"
                ]
            ),

            nn.Linear(
                model_config[
                    "classifier_hidden_dim"
                ],
                dataset_config[
                    "num_classes"
                ]
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        image
    ):

        # TEXT EMBEDDINGS

        text_emb = (
            self.text_model.get_embeddings(
                input_ids,
                attention_mask
            )
        )

        # IMAGE EMBEDDINGS

        image_emb = (
            self.image_model.get_embeddings(
                image
            )
        )

        # PROJECTIONS

        text_emb = (
            self.text_projection(
                text_emb
            )
        )

        image_emb = (
            self.image_projection(
                image_emb
            )
        )

        # CROSS MODAL FUSION

        fused = (
            self.cross_attention(
                text_emb,
                image_emb
            )
        )

        # CLASSIFICATION

        logits = self.classifier(
            fused
        )

        return logits