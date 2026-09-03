import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from importlib import import_module
from peft import LoraConfig, get_peft_model


class LateFusion(nn.Module):

    def __init__(self, model_config, dataset_config, exp_config):
        super().__init__()

        # =========================
        # LOAD CONFIGS
        # =========================
        with open(exp_config["text_model_config"]) as f:
            text_config = yaml.safe_load(f)

        with open(exp_config["image_model_config"]) as f:
            image_config = yaml.safe_load(f)

        # =========================
        # IMPORT MODELS
        # =========================
        text_module = import_module(text_config["model_import"])
        image_module = import_module(image_config["model_import"])

        TextModelClass = getattr(text_module, text_config["model_class"])
        ImageModelClass = getattr(image_module, image_config["model_class"])

        self.text_model = TextModelClass(text_config, dataset_config, exp_config)
        self.image_model = ImageModelClass(image_config, dataset_config, exp_config)

        # =========================
        # FREEZE
        # =========================
        for p in self.text_model.parameters():
            p.requires_grad = False

        for p in self.image_model.parameters():
            p.requires_grad = False

        # =========================
        # EMBEDDING DIMS (REAL FIX)
        # =========================
        self.text_dim = self._get_text_dim()
        self.image_dim = self._get_image_dim()

        fusion_dim = self.text_dim + self.image_dim

        # =========================
        # CLASSIFIER (FIXED)
        # =========================
        hidden = model_config["classifier_hidden_dim"]

        self.proj = nn.Linear(fusion_dim, hidden)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(hidden, 1 if exp_config.get("loss") == "bce"
                      else dataset_config["num_classes"])
        )

        print(f"Fusion dim: {fusion_dim}")

    # =========================
    # DIM DETECTION SAFE
    # =========================
    def _get_text_dim(self):
        return self.text_model.encoder.config.hidden_size

    def _get_image_dim(self):
        try:
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = self.image_model.encoder(dummy)

            if isinstance(out, torch.Tensor):
                if out.dim() == 4:
                    return out.mean(dim=[2, 3]).shape[-1]
                elif out.dim() == 3:
                    return out.shape[-1]
                else:
                    return out.shape[-1]

            return out.last_hidden_state.shape[-1]

        except Exception:
            raise RuntimeError("Cannot infer image embedding dim")

    # =========================
    # FORWARD
    # =========================
    def forward(self, input_ids, attention_mask, image):

        # TEXT
        text_out = self.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_out.last_hidden_state[:, 0]

        # IMAGE
        img_out = self.image_model.encoder(image)

        if hasattr(img_out, "last_hidden_state"):
            img_feat = img_out.last_hidden_state[:, 0]

        elif isinstance(img_out, torch.Tensor):
            if img_out.dim() == 4:
                img_feat = img_out.mean(dim=[2, 3])
            elif img_out.dim() == 3:
                img_feat = img_out[:, 0]
            else:
                img_feat = img_out

        else:
            raise ValueError("Unsupported image output")

        # NORMALIZATION
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)

        # =========================
        # FUSION (FIXED)
        # =========================
        fused = torch.cat([text_feat, img_feat], dim=-1)

        fused = self.proj(fused)

        logits = self.classifier(fused)

        return logits