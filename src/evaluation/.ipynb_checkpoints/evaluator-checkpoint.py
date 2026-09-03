import os
import json
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch.distributed as dist

from evaluation.metrics import compute_metrics


class Evaluator:

    def __init__(self, model, criterion, device, config, modalities):

        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        self.modalities = modalities

        self.output_dir = os.path.join(
            config["output_dir"],
            config["experiment_name"]
        ).replace("${dataset_name}", config["dataset_name"])

        os.makedirs(self.output_dir, exist_ok=True)


    def _gather(self, tensor):

        if not dist.is_available() or not dist.is_initialized():
            return tensor

        tensor = tensor.detach().to("cuda")  

        world_size = dist.get_world_size()

        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

        dist.all_gather(gather_list, tensor)

        return torch.cat(gather_list, dim=0)

    @torch.no_grad()
    def evaluate(self, dataloader, split="val"):

        self.model.eval()

        total_loss = 0

        local_preds = []
        local_labels = []

        for batch in tqdm(dataloader):

            labels = batch["label"].to(self.device)

            if self.modalities["text"] and not self.modalities["image"]:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

            elif self.modalities["image"] and not self.modalities["text"]:

                images = batch["image"].to(self.device)
                outputs = self.model(images)

            else:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images = batch["image"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=images
                )

            # LOSS + PRED
            if self.config.get("loss") == "bce":

                labels_loss = labels.float().unsqueeze(1)
                loss = self.criterion(outputs, labels_loss)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().squeeze(1)

            else:

                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

            total_loss += loss.item()

            local_preds.append(preds.detach().cpu())
            local_labels.append(labels.detach().cpu())

        local_preds = torch.cat(local_preds, dim=0)
        local_labels = torch.cat(local_labels, dim=0)

        global_preds = self._gather(local_preds)
        global_labels = self._gather(local_labels)

        global_preds = global_preds.cpu().numpy()
        global_labels = global_labels.cpu().numpy()

        metrics = compute_metrics(global_labels, global_preds)

        metrics["loss"] = total_loss / len(dataloader)

        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:

            self.save_metrics(metrics, split)

            if split == "test":
                self.save_confusion_matrix(global_labels, global_preds, split)

        return metrics

    def save_metrics(self, metrics, split):

        save_path = os.path.join(self.output_dir, f"{split}_metrics.json")

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def save_confusion_matrix(self, labels, preds, split):

        cm = confusion_matrix(labels, preds)

        class_names = self.config.get("class_names", ["0", "1"])

        fig, ax = plt.subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")

        ax.set_title(f"{split.capitalize()} Confusion Matrix",
                     fontsize=22, fontweight="bold", pad=20)

        ax.set_xlabel("Predicted Label", fontsize=18)
        ax.set_ylabel("True Label", fontsize=18)

        ax.tick_params(axis="both", labelsize=15)

        for text in disp.text_.ravel():
            text.set_fontsize(18)
            text.set_fontweight("bold")

        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir,
            f"{split}_confusion_matrix.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)