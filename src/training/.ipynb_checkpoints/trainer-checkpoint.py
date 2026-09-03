import os
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        modalities, 
        config
    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.modalities = modalities

        self.config = config

        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "train_accuracy": [],

            "val_loss": [],
            "val_accuracy": []
        }

        self.output_dir = Path(
            os.path.join(
                config["output_dir"],
                config["experiment_name"]
            ).replace("${dataset_name}", config["dataset_name"])
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        #save config - hyperparameters
        with open(
            self.output_dir / "config.yaml",
            "w"
        ) as f:

            yaml.dump(
                self.config,
                f,
                sort_keys=False
            )



    def train_epoch(
        self,
        dataloader
    ):

        self.model.train()

        total_loss = 0

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader):

            labels = batch[
                "label"
            ].to(self.device)

            inputs = {}

            if self.modalities["text"]:

                inputs["input_ids"] = batch[
                    "input_ids"
                ].to(self.device)

                inputs["attention_mask"] = batch[
                    "attention_mask"
                ].to(self.device)

            if self.modalities["image"]:

                inputs["image"] = batch[
                    "image"
                ].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                **inputs
            )

            #criterion 
            if self.config.get("loss") == "bce":

                labels = labels.float().unsqueeze(1)
            
                loss = self.criterion(
                    outputs,
                    labels
                )
            
            else:
            
                loss = self.criterion(
                    outputs,
                    labels
                )

            loss.backward()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # current_lr = self.optimizer.param_groups[0]["lr"]
            # print(f"LR: {current_lr:.8f}")

            total_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

        metrics = {

            "loss":
            total_loss / len(dataloader),

            "accuracy":
            accuracy_score(
                all_labels,
                all_preds
            ),
            
        }

        return metrics
    
    def save_model(self, name="best_model.pt"):

        # solo rank0 guarda
        if dist.is_available() and dist.is_initialized():
    
            if dist.get_rank() != 0:
                return
    
        save_path = self.output_dir / name
    
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )
    
        model_to_save = (
            self.model.module
            if isinstance(self.model, DDP)
            else self.model
        )
    
        torch.save(
            model_to_save.state_dict(),
            save_path
        )
    
        print(f"Model saved at {save_path}")
        
    def load_model(
        self,
        name="best_model.pt",
        strict=True,
        path=None
    ):
    
        load_path = (
            path
            if path is not None
            else self.output_dir / name
        )
    
        state_dict = torch.load(
            load_path,
            map_location="cpu",
            weights_only=True
        )
    
        if isinstance(self.model, (nn.DataParallel, DDP)):
    
            self.model.module.load_state_dict(
                state_dict,
                strict=strict
            )
    
        else:
    
            self.model.load_state_dict(
                state_dict,
                strict=strict
            )
    
        self.model.to(self.device)


    def update_history(
        self,
        train_metrics,
        val_metrics
    ):

        self.history["train_loss"].append(
            train_metrics["loss"]
        )

        self.history["train_accuracy"].append(
            train_metrics["accuracy"]
        )

        self.history["val_loss"].append(
            val_metrics["loss"]
        )

        self.history["val_accuracy"].append(
            val_metrics["accuracy"]
        )
    
    def save_plots(self):

        epochs = range(
            1,
            len(self.history["train_loss"]) + 1
        )

        # ======================
        # LOSS
        # ======================

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            self.history["train_loss"],
            label="Train Loss",
            linewidth=3
        )

        plt.plot(
            epochs,
            self.history["val_loss"],
            label="Validation Loss",
            linewidth=3
        )

        plt.xlabel(
            "Epoch",
            fontsize=18
        )

        plt.ylabel(
            "Loss",
            fontsize=18
        )

        plt.title(
            "Training and Validation Loss",
            fontsize=20,
            fontweight="bold"
        )

        plt.xticks(fontsize=15)

        plt.yticks(fontsize=15)

        plt.legend(
            fontsize=15
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.6
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "loss_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # ======================
        # ACCURACY
        # ======================

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            self.history["train_accuracy"],
            label="Train Accuracy",
            linewidth=3
        )

        plt.plot(
            epochs,
            self.history["val_accuracy"],
            label="Validation Accuracy",
            linewidth=3
        )

        plt.xlabel(
            "Epoch",
            fontsize=18
        )

        plt.ylabel(
            "Accuracy",
            fontsize=18
        )

        plt.title(
            "Training and Validation Accuracy",
            fontsize=20,
            fontweight="bold"
        )

        plt.xticks(fontsize=15)

        plt.yticks(fontsize=15)

        plt.legend(
            fontsize=15
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.6
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "accuracy_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()