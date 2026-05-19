import yaml

import torch
import torch.nn as nn

from importlib import import_module

from dataset.dataloader import (
    build_dataloader
)

from training.trainer import (
    Trainer
)

from dataset.processing.text import ( TextProcessor )

from evaluation.evaluator import (
    Evaluator
)

from evaluation.callbacks.optimizer import (
    build_optimizer
)


EXPERIMENT_PATH = (

    "src/config/experiments/text/e5.yaml"

)

with open(EXPERIMENT_PATH) as f:

    exp_config = yaml.safe_load(f)

with open(exp_config["dataset"]) as f:

    dataset_config = yaml.safe_load(f)

with open(exp_config["model_config"]) as f:

    model_config = yaml.safe_load(f)


print("Model: ", exp_config["experiment_name"])
print("Modalities: ", exp_config["modalities"])


device = torch.device(
    exp_config["device"]
)

model_class_name = model_config[
    "model_class"
]

model_module = import_module(
    model_config["model_import"]
)

ModelClass = getattr(
    model_module,
    model_class_name
)

model = ModelClass(
    model_config=model_config,
    dataset_config=dataset_config,
    exp_config=exp_config
)

model.to(device)


tokenizer = None

processors = {}


if exp_config["modalities"]["text"]:

    tokenizer = model.get_tokenizer()
    processors["text"] = TextProcessor()


if exp_config["modalities"]["image"]:

    if exp_config["experiment_name"].startswith("Dino"):

        processors["image"] = (
            model.get_image_processor()
        )

    else:

        processors["image"] = (
            model.get_image_transform()
        )

train_loader = build_dataloader(

    csv_path=dataset_config["train_csv"],
    dataset_config=dataset_config,
    training_config=exp_config["training"],
    processors=processors,
    modalities=exp_config["modalities"],
    tokenizer=tokenizer
)

val_loader = build_dataloader(

    csv_path=dataset_config["val_csv"],
    dataset_config=dataset_config,
    training_config=exp_config["training"],
    processors=processors,
    modalities=exp_config["modalities"],
    tokenizer=tokenizer
)

test_loader = build_dataloader(

    csv_path=dataset_config["test_csv"],
    dataset_config=dataset_config,
    training_config=exp_config["training"],
    processors=processors,
    modalities=exp_config["modalities"],
    tokenizer=tokenizer
)


criterion = nn.CrossEntropyLoss()

optimizer = build_optimizer(

    model=model,

    optimizer_config=exp_config[
        "optimizer"
    ],

    training_config=exp_config[
        "training"
    ]
)

trainer = Trainer(

    model=model,

    optimizer=optimizer,

    criterion=criterion,

    device=device,

    modalities=exp_config[
        "modalities"
    ],

    config={**exp_config}
)

evaluator = Evaluator(

    model=model,

    criterion=criterion,

    device=device,

    config={
        **exp_config,
        **dataset_config,
        **model_config
    },

    modalities=exp_config[
        "modalities"
    ]
)


epochs = exp_config["training"]["epochs"]

best_acc = 0

for epoch in range(epochs):

    print(
        f"\nEpoch {epoch+1}/{epochs}"
    )

    train_metrics = trainer.train_epoch(
        train_loader
    )

    val_metrics = evaluator.evaluate(
        val_loader,
        split="val"
    )

    print(
        f"TRAIN | "
        f"loss={train_metrics['loss']:.4f} "
        f"acc={train_metrics['accuracy']:.4f}"
    )

    print(
        f"VAL   | "
        f"loss={val_metrics['loss']:.4f} "
        f"acc={val_metrics['accuracy']:.4f}"
    )

    trainer.update_history(
        train_metrics,
        val_metrics
    )

    trainer.save_plots()

    # save best model
    if val_metrics["accuracy"] > best_acc:

        best_acc = val_metrics[
            "accuracy"
        ]

        trainer.save_model(
            "best_model.pt"
        )

        print(
            "Best model saved!"
        )



print("\nFinal Test Evaluation")

#load best model
trainer.load_model()
print("Best model loaded for testing!")



test_metrics = evaluator.evaluate(
    test_loader,
    split="test"
)

print(
    f"TEST  | "
    f"loss={test_metrics['loss']:.4f} "
    f"acc={test_metrics['accuracy']:.4f} "
    f"f1={test_metrics['f1']:.4f} "
    f"precision={test_metrics['precision']:.4f} "
    f"recall={test_metrics['recall']:.4f}"
)