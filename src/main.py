import yaml
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from importlib import import_module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset.dataloader import build_dataloader
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

from evaluation.callbacks.optimizer import build_optimizer
from evaluation.callbacks.scheduler import build_scheduler

from dataset.processing.text import TextProcessor
from dataset.processing.text_augmentation import BackTranslationAugmenter
from dataset.processing.image import train_transform

from utils.weightclass import compute_class_weights


EXPERIMENT_PATH = "src/config/experiments/multimodal/crossmm/bge_PE_core_cross.yaml"
# "src/config/experiments/multimodal/crossmm/bge_ViT_cross.yaml"

with open(EXPERIMENT_PATH) as f:
    exp_config = yaml.safe_load(f)

with open(exp_config["dataset"]) as f:
    dataset_config = yaml.safe_load(f)

with open(exp_config["model_config"]) as f:
    model_config = yaml.safe_load(f)


# =========================
# MODEL IMPORT
# =========================
model_module = import_module(model_config["model_import"])
ModelClass = getattr(model_module, model_config["model_class"])

model = ModelClass(
    model_config=model_config,
    dataset_config=dataset_config,
    exp_config=exp_config
)

base_model = model


# =========================
# DDP INIT
# =========================
use_ddp = torch.cuda.is_available() and torch.cuda.device_count() > 1

if use_ddp:

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    is_main_process = dist.get_rank() == 0

    if is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs with DDP")

else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    is_main_process = True


# =========================
# TOKENIZER / PROCESSORS
# =========================
tokenizer = None
processors = {}

if exp_config["modalities"]["text"]:
    tokenizer = (
        base_model.text_model.get_tokenizer()
        if hasattr(base_model, "text_model")
        else base_model.get_tokenizer()
    )

    processors["text_train"] = TextProcessor(
        augmenter=BackTranslationAugmenter(p=1)
        if model_config.get("text_augmentation", False)
        else None
    )
    processors["text_eval"] = TextProcessor(augmenter=None)


if exp_config["modalities"]["image"]:

    if hasattr(base_model, "image_model"):
        base_transform = (
            base_model.image_model.get_image_processor()
            if exp_config["experiment_name"].startswith("Dino")
            else base_model.image_model.get_image_transform()
        )
    else:
        base_transform = (
            base_model.get_image_processor()
            if exp_config["experiment_name"].startswith("Dino")
            else base_model.get_image_transform()
        )

    processors["image_train"] = (
        train_transform(base_transform)
        if model_config.get("image_augmentation", False)
        else base_transform
    )

    processors["image_eval"] = base_transform


# =========================
# DATA
# =========================
train_loader = build_dataloader(
    csv_path=dataset_config["train_csv"],
    dataset_config=dataset_config,
    training_config=exp_config["training"],
    processors=processors,
    modalities=exp_config["modalities"],
    tokenizer=tokenizer,
    split="train"
)

val_loader = build_dataloader(
    csv_path=dataset_config["val_csv"],
    dataset_config=dataset_config,
    training_config=exp_config["training"],
    processors=processors,
    modalities=exp_config["modalities"],
    tokenizer=tokenizer,
    split="val"
)

test_loader = build_dataloader(
    csv_path=dataset_config["test_csv"],
    dataset_config=dataset_config,
    training_config=exp_config["training"],
    processors=processors,
    modalities=exp_config["modalities"],
    tokenizer=tokenizer,
    split="test"
)


# =========================
# LOSS
# =========================
class_weights = compute_class_weights(
    csv_path=dataset_config["train_csv"],
    label_column=dataset_config["label_column"],
    device=device
)

if exp_config.get("loss") == "bce":
    pos_weight = (class_weights[0] / class_weights[1]).unsqueeze(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)


# =========================
# OPTIM / SCHED
# =========================
optimizer = build_optimizer(
    model=model,
    optimizer_config=exp_config["optimizer"],
    training_config=exp_config["training"]
)

scheduler = build_scheduler(
    optimizer=optimizer,
    scheduler_config=exp_config["scheduler"],
    training_config=exp_config["training"],
    train_dataloader=train_loader
)


# =========================
# TRAINER / EVAL
# =========================

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    modalities=exp_config["modalities"],
    config={**exp_config}
)

if exp_config.get("load_best_model", False):

    base_exp = exp_config["experiment_name"].replace(
        "_lora",
        ""
    )

    model_path = os.path.join(
        exp_config["output_dir"],
        base_exp,
        "best_model.pt"
    ).replace(
        "${dataset_name}",
        exp_config["dataset_name"]
    )

    print(f"Loading pretrained model: {model_path}")

    trainer.load_model(
        strict=False,
        path=model_path
    )

evaluator = Evaluator(
    model=model,
    criterion=criterion,
    device=device,
    config={**exp_config, **dataset_config, **model_config},
    modalities=exp_config["modalities"]
)


# =========================
# LOG ONLY MAIN
# =========================
def log(*args, **kwargs):
    if is_main_process:
        print(*args, **kwargs)


log("Model:", exp_config["experiment_name"])
log("Dataset:", exp_config["dataset_name"])
log("Modalities:", exp_config["modalities"])


# =========================
# TRAIN LOOP
# =========================
epochs = exp_config["training"]["epochs"]
best_f1 = 0

for epoch in range(epochs):

    if use_ddp and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    train_metrics = trainer.train_epoch(train_loader)
    val_metrics = evaluator.evaluate(val_loader, split="val")

    if is_main_process:

        print(f"\nEpoch {epoch+1}/{epochs}")

        print(f"TRAIN | loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f}")
        print(f"VAL   | loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")

        trainer.update_history(train_metrics, val_metrics)
        trainer.save_plots()

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            trainer.save_model("best_model.pt")
            print("Best model saved!")

    if use_ddp:
        dist.barrier()


# =========================
# TEST
# =========================
if use_ddp:
    dist.barrier()

if is_main_process:
    trainer.load_model("best_model.pt" if best_f1 > 0 else None)

if use_ddp:
    dist.barrier()

log(f"[Rank {dist.get_rank() if use_ddp else 0}] Starting test")

test_metrics = evaluator.evaluate(test_loader, split="test")

if is_main_process:
    print("\nFinal Test Evaluation")
    print(
        f"TEST | loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )

if use_ddp:
    dist.destroy_process_group()