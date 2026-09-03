from platform import processor

import pandas as pd

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import random
from collections import defaultdict

class BaseDataset(Dataset):

    def __init__(
        self,
        csv_path,
        config,
        processors,
        modalities,
        tokenizer=None,
        split="train"
    ):

        self.df = pd.read_csv(csv_path)

        self.config = config

        self.processors = processors

        self.modalities = modalities

        self.tokenizer = tokenizer

        self.split = split

        self.use_sse = config.get(
            "use_sse",
            False
        )
        
        self.sse_prob = config.get(
            "sse_prob",
            0.2
        )
        
        self.sse_same_class_prob = config.get(
            "sse_same_class_prob",
            0.9
        )

        self.class_indices = {}
        self.labels = []

        if self.use_sse and self.split == "train":

            self.class_indices = defaultdict(list)
        
            for idx, row in self.df.iterrows():
        
                label = int(
                    row[self.config["label_column"]]
                )
        
                self.class_indices[label].append(idx)
        
            self.labels = sorted(
                self.class_indices.keys()
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        #sse
        if self.use_sse and self.split == "train":
        
            label = int(
                row[self.config["label_column"]]
            )
        
            # ============
            # IMAGE SSE
            # ============
        
            if self.modalities["image"]:
        
                if self._should_do(self.sse_prob):
        
                    if self._should_do(
                        self.sse_same_class_prob
                    ):
        
                        target_idx = self._sample_same_class(
                            label,
                            idx
                        )
        
                    else:
        
                        target_idx = self._sample_other_class(
                            label
                        )
        
                    row = row.copy()
        
                    row[
                        self.config["image_column"]
                    ] = self.df.iloc[target_idx][
                        self.config["image_column"]
                    ]
        
            # ============
            # TEXT SSE
            # ============
        
            if self.modalities["text"]:
        
                if self._should_do(self.sse_prob):
        
                    if self._should_do(
                        self.sse_same_class_prob
                    ):
        
                        target_idx = self._sample_same_class(
                            label,
                            idx
                        )
        
                    else:
        
                        target_idx = self._sample_other_class(
                            label
                        )
        
                    row = row.copy()
        
                    row[
                        self.config["text_column"]
                    ] = self.df.iloc[target_idx][
                        self.config["text_column"]
                    ]

        sample = {}

        if self.modalities["text"]:

            text = str(
                row[self.config["text_column"]]
            )


            if self.split == "train":
                text_processor = self.processors.get("text_train", None)
            else:
                text_processor = self.processors.get("text_eval", None)
        
            if text_processor is not None:
                text = text_processor(text)
                sample["raw_text"] = text
        
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt"
            )

            sample["input_ids"] = encoded[
                "input_ids"
            ].squeeze(0)

            sample["attention_mask"] = encoded[
                "attention_mask"
            ].squeeze(0)

        if self.modalities["image"]:

            image_path = row[
                self.config["image_column"]
            ]
        
            image = Image.open(
                image_path
            ).convert("RGB")
        
            if self.split == "train":
        
                processor = self.processors.get(
                    "image_train"
                )
        
            else:
        
                processor = self.processors.get(
                    "image_eval"
                )
        
            if processor is not None:
        
                # HuggingFace processor
                if hasattr(processor, "__class__") and (
                    "ImageProcessor" in processor.__class__.__name__
                ):
        
                    image = processor(
                        images=image,
                        return_tensors="pt"
                    )["pixel_values"].squeeze(0)
        
                # torchvision transforms
                else:
        
                    image = processor(image)
        
            sample["image"] = image
            sample["image_path"] = image_path

        sample["label"] = int(
            row[self.config["label_column"]]
        )

        sample["label_text"] = str(
            row[self.config[
                "label_text_column"
            ]]
        )

        return sample

    def _should_do(self, p):
    
        return random.random() < p
    
    
    def _sample_same_class(
        self,
        label,
        current_idx
    ):
    
        candidates = [
            i
            for i in self.class_indices[label]
            if i != current_idx
        ]
    
        if len(candidates) == 0:
    
            return current_idx
    
        return random.choice(candidates)
    
    
    def _sample_other_class(
        self,
        label
    ):
    
        other_labels = [
            l
            for l in self.labels
            if l != label
        ]
    
        target_label = random.choice(
            other_labels
        )
    
        return random.choice(
            self.class_indices[target_label]
        )