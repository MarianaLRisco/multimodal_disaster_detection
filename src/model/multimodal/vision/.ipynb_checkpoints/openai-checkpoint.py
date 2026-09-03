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
                    
        self.embedding_dim = model_config["embedding_dim"]

        self.classifier = nn.Sequential(

            nn.Linear(
                self.embedding_dim,
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

    def get_embeddings(
        self,
        image
    ):

        embeddings = self.encoder(image)

        return embeddings

    def forward(
        self,
        image
    ):

        embeddings = self.get_embeddings(
            image
        )

        logits = self.classifier(
            embeddings
        )

        return logits

    def get_image_transform(self):

        return self.image_transform
    
    def get_patch_tokens(
        self,
        image,
        return_cls=False
    ):

        encoder = self.encoder

        if hasattr(encoder, "transformer"):

            x = encoder.conv1(image)

            x = x.reshape(
                x.shape[0],
                x.shape[1],
                -1
            )

            x = x.permute(0, 2, 1)

            cls_token = encoder.class_embedding.to(
                x.dtype
            )

            cls_tokens = cls_token.unsqueeze(0).unsqueeze(0)

            cls_tokens = cls_tokens.expand(
                x.shape[0],
                -1,
                -1
            )

            x = torch.cat(
                [cls_tokens, x],
                dim=1
            )

            x = x + encoder.positional_embedding.to(
                x.dtype
            )

            x = encoder.patch_dropout(x)

            x = encoder.ln_pre(x)

            x = x.permute(1, 0, 2)

            x = encoder.transformer(x)

            x = x.permute(1, 0, 2)

            if hasattr(encoder, "ln_post"):

                x = encoder.ln_post(x)

            cls_embedding = x[:, 0, :]

            patch_tokens = x[:, 1:, :]

            if return_cls:

                return cls_embedding, patch_tokens

            return patch_tokens

        else:

            raise ValueError(
                "Backbone does not support patch extraction"
            )
