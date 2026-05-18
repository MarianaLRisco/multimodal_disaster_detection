import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel
)

from utils.pooling import (
    mean_pooling,
    cls_pooling,
    last_token_pooling
)


class E5(nn.Module):

    def __init__(self, model_config,
        dataset_config):

        super().__init__()

        self.model_config = model_config
        self.dataset_config = dataset_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["hf_name"]
        )

        self.encoder = AutoModel.from_pretrained(
            model_config["hf_name"]
        )

        if model_config["freeze"]:

            for param in self.encoder.parameters():
                param.requires_grad = False
        
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

        self.pooling = model_config["pooling"]

        self.classifier = nn.Sequential(

            nn.Linear(
                model_config["embedding_dim"],
                model_config["hidden_dim"]
            ),

            nn.ReLU(),

            nn.Dropout(
                model_config["dropout"]
            ),

            nn.LayerNorm(
                model_config["hidden_dim"],
            ),

            nn.Linear(
                model_config["hidden_dim"],
                dataset_config["num_classes"]
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if self.pooling == "mean":

            embeddings = mean_pooling(
                outputs,
                attention_mask
            )

        elif self.pooling == "cls":

            embeddings = cls_pooling(
                outputs
            )

        elif self.pooling == "last":

            embeddings = last_token_pooling(
                outputs,
                attention_mask
            )

        else:

            raise ValueError(
                f"Unknown pooling: {self.pooling}"
            )
        
        # if self.model_config.get("model_class", None) == "E5":

        if self.model_config.get("normalize_embeddings",True):
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

        logits = self.classifier(
            embeddings
        )

        return logits

    def get_tokenizer(self):

        return { 'tokenizer': self.tokenizer}

    def get_model(self):

        return self.encoder