import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from importlib import import_module
from utils.crossmm import CrossAttentionFusion

#LoRA
from peft import LoraConfig, get_peft_model 

class CrossModalTransformerModel(nn.Module):

    def __init__(
        self,
        model_config,
        dataset_config,
        exp_config
    ):

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

        text_module = import_module(
            text_config["model_import"]
        )

        image_module = import_module(
            image_config["model_import"]
        )

        TextModelClass = getattr(
            text_module,
            text_config["model_class"]
        )

        ImageModelClass = getattr(
            image_module,
            image_config["model_class"]
        )


        # BACKBONES
        # =========================

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

        # for name, module in self.image_model.encoder.named_modules():
        #     print(name)


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

       
        # =========================
        # DIMS
        # =========================

        self.text_dim = text_config[
            "embedding_dim"
        ]

        self.image_dim = image_config[
            "embedding_dim"
        ]

        self.hidden_dim = model_config[
            "cross_hidden_dim"
        ]

    
        # =========================
        # CROSS MODAL FUSION
        # =========================

        self.cross_attention = CrossAttentionFusion(

            text_dim=self.text_dim,

            image_dim=self.image_dim,

            hidden_dim=self.hidden_dim,

            num_heads=model_config.get(
                "num_heads",
                8
            )
        )

   

        # =========================
        # CLASSIFIER
        # =========================

        output_dim = (
            1
            if exp_config.get("loss") == "bce"
            else dataset_config["num_classes"]
        )


        self.gate_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_i = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.classifier = nn.Sequential(

            nn.Linear(
                self.hidden_dim,
                model_config[
                    "classifier_hidden_dim"
                ]
            ),

            nn.GELU(),

            nn.Dropout(
                model_config["dropout"]
            ),

            nn.Linear(
                model_config[
                    "classifier_hidden_dim"
                ],
                output_dim
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
        input_ids,
        attention_mask,
        image
    ):

        # =========================
        # TEXT TOKENS
        # =========================

        text_outputs = self.text_model.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        text_tokens = (
            text_outputs.last_hidden_state
        )

        # IMAGE TOKENS
        encoder = self.image_model.encoder

        if hasattr(encoder, "trunk"):
            image_outputs = encoder.trunk.forward_features(image)
        else:
            image_outputs =  self.image_model.get_patch_tokens(image)
            
            # CASE 1: ViT / PE-core (tokens)
            
        if hasattr(image_outputs, "last_hidden_state"):
                image_tokens = image_outputs.last_hidden_state

        # CASE 2: raw tensor output
        elif isinstance(image_outputs, torch.Tensor):
        
            if image_outputs.dim() == 4:
                B, C, H, W = image_outputs.shape
                image_tokens = image_outputs.flatten(2).transpose(1, 2)
            
            elif image_outputs.dim() == 3:
                image_tokens = image_outputs
            else:
                raise ValueError(f"Unsupported shape: {image_outputs.shape}")
        
        else:
            raise ValueError("Unsupported image encoder output")

        # text_tokens = text_tokens[:, :5, :]
        # image_tokens =  image_tokens[:,:67,:]

        # print(text_tokens)
        # print(image_tokens)

        text_tokens = text_tokens.float()
        image_tokens = image_tokens.float()

        fused = self.cross_attention(text_tokens, image_tokens)
        
        logits = self.classifier(fused)
    
        return logits