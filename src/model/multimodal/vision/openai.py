import torch
import torch.nn as nn

import open_clip


class openaiModels(nn.Module):

    def __init__(
        self,
        model_config,
        dataset_config,
        exp_config
    ):

        super().__init__()

        self.model_config = model_config
        self.dataset_config = dataset_config
        self.exp_config = exp_config

        self.encoder, _, self.image_transform = (
            open_clip.create_model_and_transforms(

                model_name=model_config[
                    "hf_name"
                ],

                pretrained=model_config[
                    "pretrained"
                ]
            )
        )

        self.encoder = (
            self.encoder.visual
        )

        if model_config["freeze"]:

            for param in self.encoder.parameters():

                param.requires_grad = False

        n_layers = exp_config.get(
            "unfreeze_last_n_layers",
            0
        )

        if n_layers > 0:

            children = list(
                self.encoder.children()
            )

            for layer in children[-n_layers:]:

                for param in layer.parameters():

                    param.requires_grad = True

        self.classifier = nn.Sequential(

            nn.Linear(
                model_config["embedding_dim"],
                model_config["hidden_dim"]
            ),

            nn.ReLU(),

            nn.Dropout(
                model_config["dropout"]
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

        embeddings = self.encoder(
            image
        )

        logits = self.classifier(
            embeddings
        )

        return logits

    def get_image_transform(self):

        return self.image_transform