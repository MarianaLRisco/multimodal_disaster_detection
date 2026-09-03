import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from utils.crossAttn import CrossAttentionFusion
from importlib import import_module
from peft import LoraConfig, get_peft_model 

class CrossModel(nn.Module):

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

       
        # =========================
        # FREEZE BACKBONES
        # =========================
        
        for param in self.text_model.parameters():
        
            param.requires_grad = False
        
        for param in self.image_model.parameters():
        
            param.requires_grad = False
        
        # =========================
        # OPTIONAL LORA
        # =========================
        
        use_lora = exp_config.get(
            "use_lora",
            False
        )
        
        if use_lora:
        
            lora_cfg = exp_config["lora"]
        
            if lora_cfg.get("text", False):

                text_layers = (
                    self.text_model.encoder.encoder.layer
                )
            
                n_text = lora_cfg["last_n_text_layers"]
            
                last_text_layers = text_layers[-n_text:]
            
                for layer in last_text_layers:
            
                    text_lora_config = LoraConfig(
            
                        r=lora_cfg["r"],
            
                        lora_alpha=lora_cfg["alpha"],
            
                        target_modules=[
                            "query",
                            "value"
                        ],
            
                        lora_dropout=lora_cfg["dropout"],
            
                        bias="none"
                    )
            
                    layer.attention.self = get_peft_model(
                        layer.attention.self,
                        text_lora_config
                    )
            
                print(f"LoRA applied to LAST {n_text} TEXT layers ")
                

            # =====================
            # VISION LORA
            # =====================
            
            if lora_cfg.get("vision", False):
            
                try:
            
                    image_layers = None
            
                
                    if hasattr(self.image_model.encoder, "trunk"):

                        trunk = self.image_model.encoder.trunk
            
                        # PE-Core 
                        if hasattr(trunk, "blocks"):
            
                            image_layers = trunk.blocks
            
                        # ConvNeXt stages
                        elif hasattr(trunk, "stages"):
            
                            image_layers = []
            
                            for stage in trunk.stages:
            
                                if hasattr(stage, "blocks"):
            
                                    image_layers.extend(stage.blocks)
                    #VIT
                    elif hasattr(self.image_model.encoder, "transformer"):

                        if hasattr(
                            self.image_model.encoder.transformer,
                            "resblocks"
                        ):

                            image_layers = (
                                self.image_model.encoder
                                .transformer.resblocks
                            )
                    # =====================
                    # HuggingFace ViT style
                    # =====================
            
                    elif hasattr(
                        self.image_model.encoder,
                        "encoder"
                    ):
            
                        if hasattr(
                            self.image_model.encoder.encoder,
                            "layer"
                        ):
            
                            image_layers = (
                                self.image_model.encoder
                                .encoder.layer
                            )
            
                    # =====================
                    # PEFT safe unwrap
                    # =====================
            
                    elif hasattr(
                        self.image_model.encoder,
                        "base_model"
                    ):
            
                        base = (
                            self.image_model.encoder.base_model
                        )
            
                        if hasattr(base, "blocks"):
            
                            image_layers = base.blocks
            
                        elif hasattr(base, "encoder"):
            
                            if hasattr(base.encoder, "layer"):
            
                                image_layers = (
                                    base.encoder.layer
                                )
            
                    # =====================
                    # APPLY LORA ONLY TO LAST LAYERS
                    # =====================
            
                    if image_layers is not None:
            
                        n_last = lora_cfg[
                            "last_n_vision_layers"
                        ]
            
                        start_idx = (
                            len(image_layers) - n_last
                        )
            
                        for i in range(
                            start_idx,
                            len(image_layers)
                        ):
            
                            vision_lora_config = LoraConfig(
            
                                r=lora_cfg["r"],
            
                                lora_alpha=lora_cfg["alpha"],
            
                                target_modules=lora_cfg[
                                    "target_modules"
                                ],
            
                                lora_dropout=lora_cfg[
                                    "dropout"
                                ],
            
                                bias="none"
                            )
            
                            image_layers[i] = (
                                get_peft_model(
                                    image_layers[i],
                                    vision_lora_config
                                )
                            )
            
                        print(
                            f"LoRA applied to LAST {n_last} VISION layers"
                        )
            
                    else:
            
                        print(
                            "Could not identify vision layers for LoRA"
                        )
            
                except Exception as e:
            
                    print(
                        "Vision LoRA failed:",
                        e
                    )
            
            # =========================
            # OPTIONAL UNFREEZE
            # =========================
            
            n_layers = exp_config.get(
                "unfreeze_last_n_layers",
                0
            )
            
            if n_layers > 0:
        
                # =====================
                # TEXT
                # =====================
            
                try:
            
                    text_layers = (
                        self.text_model.encoder.encoder.layer
                    )
            
                except:
            
                    try:
            
                        text_layers = (
                            self.text_model.encoder.base_model.encoder.layer
                        )
            
                    except:
            
                        text_layers = None
            
                if text_layers is not None:
            
                    for layer in text_layers[-n_layers:]:
            
                        for param in layer.parameters():
            
                            param.requires_grad = True
            
                # =====================
                # IMAGE
                # =====================
    
    
                image_layers = None
    
                if hasattr(self.image_model.encoder, "trunk"):

                    trunk = self.image_model.encoder.trunk
        
                    # PE-Core / ViT-like inside trunk.blocks
                    if hasattr(trunk, "blocks"):
        
                        image_layers = trunk.blocks
        
                    # ConvNeXt stages
                    elif hasattr(trunk, "stages"):
        
                        image_layers = []
        
                        for stage in trunk.stages:
        
                            if hasattr(stage, "blocks"):
        
                                image_layers.extend(stage.blocks)
                #VIT
                elif hasattr(self.image_model.encoder, "transformer"):

                    if hasattr(
                        self.image_model.encoder.transformer,
                        "resblocks"
                    ):
        
                        image_layers = (
                            self.image_model.encoder
                            .transformer.resblocks
                        )
            
                # PEFT wrapper safe unwrap
                elif hasattr(self.image_model.encoder, "base_model"):
                    base = self.image_model.encoder.base_model
                    if hasattr(base, "blocks"):
                        image_layers = base.blocks
            
                if image_layers is not None:
            
                    for layer in image_layers[-n_layers:]:
            
                        for param in layer.parameters():
            
                            param.requires_grad = True


        self.text_dim = text_config["embedding_dim"]
        self.image_dim = image_config["embedding_dim"]
        self.projection_dim = model_config["projection_dim"]

        self.fusion = CrossAttentionFusion(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            proj_dim=self.projection_dim
        )

        fusion_dim = self.projection_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, model_config["classifier_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(model_config["dropout"]),
            nn.LayerNorm(model_config["classifier_hidden_dim"]),
            nn.Linear(model_config["classifier_hidden_dim"], dataset_config["num_classes"])
        )

    def forward(self, input_ids, attention_mask, image):


        text_emb = self.text_model.get_embeddings(
            input_ids,
            attention_mask
        )
    
        image_emb = self.image_model.get_embeddings(
            image
        )

        # print("text:", text_emb.shape)
        # print("image:", image_emb.shape)
    
        # text_emb = text_emb.float()
        # image_emb = image_emb.float()

        # text_emb = self.text_projection(text_emb)
        # image_emb = self.image_projection(image_emb)
    
        fused = self.fusion(text_emb, image_emb)
 
        logits = self.classifier(
            fused
        )
    
        return logits