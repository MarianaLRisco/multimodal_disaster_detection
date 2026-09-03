import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from utils.pooling import (
    mean_pooling,
    cls_pooling,
    last_token_pooling
)

from utils.attn import (
    Attention
)


class EmbeddingDecoder(nn.Module):

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

        self.attention = model_config.get(
            "attention",
            False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["hf_name"],
            trust_remote_code=True,
            use_fast=False
        )
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_config["hf_name"],
            trust_remote_code=True
        )

        if model_config.get(
            "freeze",
            False
        ):

            for param in self.encoder.parameters():

                param.requires_grad = False

        n_layers = exp_config.get(
            "unfreeze_last_n_layers",
            0
        )

        if n_layers > 0:

            try:

                transformer_layers = (
                    self.encoder.model.layers
                )

            except:

                try:

                    transformer_layers = (
                        self.encoder.bert.encoder.layer
                    )

                except:

                    transformer_layers = None

            if transformer_layers is not None:

                for layer in transformer_layers[-n_layers:]:

                    for param in layer.parameters():

                        param.requires_grad = True

        self.pooling = model_config["pooling"]

        self.embedding_dim = model_config["embedding_dim"]

        if self.attention:

            self.attn = Attention(

                feature_dim=self.embedding_dim,

                attention_dim=model_config.get(
                    "attention_dim",
                    self.embedding_dim
                )
            )

        self.classifier = nn.Sequential(

            nn.Linear(
                self.embedding_dim,
                model_config["hidden_dim"]
            ),

            nn.GELU(),

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

    def get_embeddings(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        token_embeddings = (
            outputs.hidden_states[-1]
        )

        if self.attention:

            hidden = (
                token_embeddings[:, 0]
            )

            embeddings, attention_weights = (
                self.attn(
                    token_embeddings,
                    hidden
                )
            )

        else:

            if self.pooling == "mean":

                embeddings = mean_pooling(
                    type(
                        "obj",
                        (object,),
                        {
                            "last_hidden_state":
                            token_embeddings
                        }
                    ),
                    attention_mask
                )

            elif self.pooling == "cls":

                embeddings = token_embeddings[:, 0]

            elif self.pooling == "last":

                embeddings = last_token_pooling(
                    type(
                        "obj",
                        (object,),
                        {
                            "last_hidden_state":
                            token_embeddings
                        }
                    ),
                    attention_mask
                )

            else:

                raise ValueError(
                    f"Unknown pooling: {self.pooling}"
                )

        if self.model_config.get(
            "normalize_embeddings",
            False
        ):

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        embeddings = self.get_embeddings(
            input_ids,
            attention_mask
        )

        logits = self.classifier(
            embeddings
        )

        return logits

    def get_tokenizer(self):

        return self.tokenizer

    def get_model(self):

        return self.encoder