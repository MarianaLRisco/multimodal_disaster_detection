from platform import processor

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(
        self,
        csv_path,
        config,
        processors,
        modalities,
        tokenizer=None
    ):

        self.df = pd.read_csv(csv_path)

        self.config = config

        self.processors = processors

        self.modalities = modalities

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample = {}

        if self.modalities["text"]:

            text = str(
                row[self.config["text_column"]]
            )

            if "text" in self.processors:

                processor = self.processors[
                    "text"
                ]

                if processor is not None:

                    text = processor(text)

            encoded = self.tokenizer(

                text,

                padding="max_length",

                truncation=True,

                max_length=self.config[
                    "max_length"
                ],

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

            if "image" in self.processors:

                processor = self.processors[
                    "image"
                ]

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

        sample["label"] = int(
            row[self.config["label_column"]]
        )

        sample["label_text"] = str(
            row[self.config[
                "label_text_column"
            ]]
        )

        return sample