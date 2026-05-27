from torch.utils.data import DataLoader

from dataset.base_dataset import (
    BaseDataset
)


def build_dataloader(
    csv_path,
    dataset_config,
    training_config,
    processors,
    modalities,
    tokenizer=None,
    split="train"
):

    dataset = BaseDataset(

        csv_path=csv_path,

        config=dataset_config,

        processors=processors,

        modalities=modalities,

        tokenizer=tokenizer,

        split=split
    )

    dataloader = DataLoader(

        dataset,

        batch_size=training_config[
            "batch_size"
        ],

        shuffle=training_config[
            "shuffle"
        ],

        num_workers=training_config[
            "num_workers"
        ],

        pin_memory=True
    )

    return dataloader