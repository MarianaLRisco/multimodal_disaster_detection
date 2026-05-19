import torch
import torch.nn as nn

from transformers import (
    AutoImageProcessor,
    AutoModel
)


class Dino(nn.Module):

    def __init__(
        self,
        model_config,
        dataset_config
    ):

        super().__init__()

        self.model_config = model_config
        self.dataset_config = dataset_config

        self.processor = (
            AutoImageProcessor.from_pretrained(
                model_config["hf_name"]
            )
        )

        self.encoder = (
            AutoModel.from_pretrained(
                model_config["hf_name"]
            )
        )

        # freeze encoder
        if model_config["freeze"]:

            for param in self.encoder.parameters():

                param.requires_grad = False

        # unfreeze last n layers
        n_layers = model_config.get(
            "unfreeze_last_n_layers",
            0
        )

        if n_layers > 0:

            transformer_layers = (
                self.encoder.encoder.layer
            )

            for layer in transformer_layers[-n_layers:]:

                for param in layer.parameters():

                    param.requires_grad = True

        # hidden size automatically
        self.embedding_dim = (
            self.encoder.config.hidden_size
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                self.embedding_dim,
                model_config["hidden_dim"]
            ),

            nn.ReLU(),

            nn.Dropout(
                model_config["dropout"]
            ),

            nn.LayerNorm(
                model_config["hidden_dim"]
            ),

            nn.Linear(
                model_config["hidden_dim"],
                dataset_config["num_classes"]
            )
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        print(
            f"Trainable params: "
            f"{trainable:,}/{total:,}"
        )

    def forward(
        self,
        image
    ):

        outputs = self.encoder(
            pixel_values=image
        )

        # CLS token
        embeddings = (
            outputs.last_hidden_state[:, 0]
        )

        logits = self.classifier(
            embeddings
        )

        return logits

    def get_image_processor(self):

        return self.processor