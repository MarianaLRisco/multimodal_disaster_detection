import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from utils.crossmodalAttn import CrossModalAttention
from importlib import import_module


class CrossModalModel(nn.Module):

    def __init__(
        self,
        model_config,
        dataset_config,
        exp_config
    ):

        super().__init__()

        with open(exp_config["text_model_config"]) as f:
            text_config = yaml.safe_load(f)

        with open(exp_config["image_model_config"]) as f:
            image_config = yaml.safe_load(f)

        text_module = import_module(text_config["model_import"])
        image_module = import_module(image_config["model_import"])

        TextModelClass = getattr(text_module, text_config["model_class"])
        ImageModelClass = getattr(image_module, image_config["model_class"])

        self.text_model = TextModelClass(
            text_config,
            dataset_config,
            exp_config
        )

        self.image_model = ImageModelClass(
            image_config,
            dataset_config,
            exp_config
        )

        self.model_config = model_config
        self.dataset_config = dataset_config
        self.exp_config = exp_config

        n_layers = exp_config.get("unfreeze_last_n_layers", 0)

        if n_layers > 0:

            try:
                text_layers = self.text_model.encoder.encoder.layer
            except:
                try:
                    text_layers = self.text_model.encoder.transformer.layer
                except:
                    text_layers = None

            if text_layers is not None:
                for layer in text_layers[-n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            try:
                image_layers = self.image_model.encoder.encoder.layer
            except:
                try:
                    image_layers = self.image_model.encoder.layers
                except:
                    try:
                        image_layers = self.image_model.encoder.blocks
                    except:
                        image_layers = None

            if image_layers is not None:
                for layer in image_layers[-n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.text_dim = text_config["embedding_dim"]
        self.image_dim = image_config["embedding_dim"]
        self.projection_dim = model_config["projection_dim"]

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, self.projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim, self.projection_dim)
        )

        self.image_projection = nn.Sequential(
            nn.Linear(self.image_dim, self.projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim, self.projection_dim)
        )

        self.cross_attention = CrossModalAttention(
            embedding_dim=self.projection_dim,
            hidden_dim=model_config["cross_hidden_dim"]
        )

        fusion_dim = model_config["cross_hidden_dim"] * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, model_config["classifier_hidden_dim"]),
            nn.GELU(),
            nn.Dropout(model_config["dropout"]),
            nn.LayerNorm(model_config["classifier_hidden_dim"]),
            nn.Linear(model_config["classifier_hidden_dim"], dataset_config["num_classes"])
        )

    def forward(self, input_ids, attention_mask, image):

        text_emb = self.text_model.get_embeddings(input_ids, attention_mask)
        image_emb = self.image_model.get_embeddings(image)

        text_emb = self.text_projection(text_emb)
        image_emb = self.image_projection(image_emb)

        # text_emb = F.normalize(text_emb, dim=-1)
        # image_emb = F.normalize(image_emb, dim=-1)

        fused = self.cross_attention(text_emb, image_emb)

        logits = self.classifier(fused)

        return logits