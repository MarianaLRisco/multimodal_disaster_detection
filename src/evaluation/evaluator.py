import os
import json

import torch

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from evaluation.metrics import (
    compute_metrics
)


class Evaluator:

    def __init__(
        self,
        model,
        criterion,
        device,
        config,
        modalities
    ):

        self.model = model

        self.criterion = criterion

        self.device = device

        self.config = config

        self.modalities = modalities

        self.output_dir = os.path.join(
            config["output_dir"],
            config["experiment_name"]
        ).replace("${dataset_name}", config["dataset_name"])

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        split="val"
    ):

        self.model.eval()

        total_loss = 0

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader):

            labels = batch[
                "label"
            ].to(self.device)

            # ======================
            # TEXT ONLY
            # ======================

            if (
                self.modalities["text"]
                and not self.modalities["image"]
            ):

                input_ids = batch[
                    "input_ids"
                ].to(self.device)

                attention_mask = batch[
                    "attention_mask"
                ].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask
                )

            # ======================
            # IMAGE ONLY
            # ======================

            elif (
                self.modalities["image"]
                and not self.modalities["text"]
            ):

                images = batch[
                    "image"
                ].to(self.device)

                outputs = self.model(
                    images
                )

            # ======================
            # MULTIMODAL
            # ======================

            elif (
                self.modalities["text"]
                and self.modalities["image"]
            ):

                input_ids = batch[
                    "input_ids"
                ].to(self.device)

                attention_mask = batch[
                    "attention_mask"
                ].to(self.device)

                images = batch[
                    "image"
                ].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=images
                )

            else:

                raise ValueError(
                    "No modality selected"
                )

            loss = self.criterion(
                outputs,
                labels
            )

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

        metrics = compute_metrics(
            all_labels,
            all_preds
        )

        metrics["loss"] = (
            total_loss / len(dataloader)
        )

        self.save_metrics(
            metrics,
            split
        )

        # ONLY TEST CONFUSION MATRIX
        if split == "test":

            self.save_confusion_matrix(
                all_labels,
                all_preds,
                split
            )

        return metrics

    def save_metrics(
        self,
        metrics,
        split
    ):

        save_path = os.path.join(
            self.output_dir,
            f"{split}_metrics.json"
        )

        with open(save_path, "w") as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

    def save_confusion_matrix(
        self,
        labels,
        preds,
        split
    ):

        cm = confusion_matrix(
            labels,
            preds
        )

        class_names = self.config.get(
            "class_names",
            ["0", "1"]
        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        disp.plot(

            ax=ax,

            cmap="Blues",

            colorbar=True,

            values_format="d"
        )

        # title
        ax.set_title(
            f"{split.capitalize()} Confusion Matrix",
            fontsize=22,
            fontweight="bold",
            pad=20
        )

        # axis labels
        ax.set_xlabel(
            "Predicted Label",
            fontsize=18
        )

        ax.set_ylabel(
            "True Label",
            fontsize=18
        )

        # tick labels
        ax.tick_params(
            axis="both",
            labelsize=15
        )

        # numbers inside cells
        for text in disp.text_.ravel():

            text.set_fontsize(18)

            text.set_fontweight("bold")

        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir,
            f"{split}_confusion_matrix.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)