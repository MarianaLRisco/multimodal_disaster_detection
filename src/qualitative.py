import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from PIL import Image
from importlib import import_module

from dataset.dataloader import build_dataloader
from dataset.processing.text import TextProcessor


# -------------------------
def load_experiment(config_path):
    with open(config_path) as f:
        exp_config = yaml.safe_load(f)

    with open(exp_config["dataset"]) as f:
        dataset_config = yaml.safe_load(f)

    with open(exp_config["model_config"]) as f:
        model_config = yaml.safe_load(f)

    return exp_config, dataset_config, model_config


# -------------------------
def build_model(exp_config, dataset_config, model_config, device):

    model_module = import_module(model_config["model_import"])
    ModelClass = getattr(model_module, model_config["model_class"])

    model = ModelClass(
        model_config=model_config,
        dataset_config=dataset_config,
        exp_config=exp_config
    )

    return model.to(device)


# -------------------------
def load_checkpoint(model, exp_config):

    model_path = os.path.join(
        exp_config["output_dir"],
        exp_config["experiment_name"],
        "best_model.pt"
    ).replace("${dataset_name}", exp_config["dataset_name"])

    print("Checkpoint:", model_path)

    ckpt = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model.load_state_dict(ckpt, strict=False)
    return model


# -------------------------
def get_predictions(model, batch, modalities):

    device = next(model.parameters()).device

    with torch.no_grad():

        images = batch["image"]

        if modalities["text"] and not modalities["image"]:

            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )

        elif modalities["image"] and not modalities["text"]:

            outputs_list = []

            for img in images:

                if isinstance(img, Image.Image):
                    img = model.image_model.get_image_transform()(img)

                img = img.to(device)

                outputs_list.append(model(img.unsqueeze(0)))

            outputs = torch.cat(outputs_list, dim=0)

        else:

            outputs_list = []

            for i in range(len(images)):

                img = images[i]

                if isinstance(img, Image.Image):
                    img = model.image_model.get_image_transform()(img)

                img = img.to(device)

                out = model(
                    input_ids=batch["input_ids"][i:i+1].to(device),
                    attention_mask=batch["attention_mask"][i:i+1].to(device),
                    image=img.unsqueeze(0)
                )

                outputs_list.append(out)

            outputs = torch.cat(outputs_list, dim=0)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

    return preds, probs.max(dim=1)[0]

# -------------------------
def save_sample(image_path, gt, pred, prob, text, save_path):

    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")

    ax.set_title(
        f"GT: {gt} \n PRED: {pred} |  P={prob:.3f}",
        fontsize=26
    )

    if text is not None:
        wrapped_text = textwrap.fill(
            text[:200],      # 
            width=40         # 
        )
    
        ax.text(
            0.5,
            -0.08,
            wrapped_text,
            fontsize=18,
            ha="center",
            va="top",
            transform=ax.transAxes,
            wrap=True
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# -------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    args = parser.parse_args()

    exp_config, dataset_config, model_config = load_experiment(args.config)

    # -------------------------
    # LABEL MAP
    # -------------------------
    df = pd.read_csv(dataset_config["test_csv"])

    label_map = dict(zip(
        df[dataset_config["label_column"]],
        df[dataset_config["label_text_column"]]
    ))

    print("LABEL MAP:", label_map)

    # -------------------------
    # DEVICE
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # MODEL
    # -------------------------
    model = build_model(exp_config, dataset_config, model_config, device)
    model = load_checkpoint(model, exp_config)
    model.eval()

    # -------------------------
    # PROCESSORS
    # -------------------------
    tokenizer = None
    processors = {}

    if exp_config["modalities"]["text"]:
        tokenizer = (
            model.text_model.get_tokenizer()
            if hasattr(model, "text_model")
            else model.get_tokenizer()
        )
        processors["text_eval"] = TextProcessor(augmenter=None)

    # -------------------------
    # DATA
    # -------------------------
    test_loader = build_dataloader(
        csv_path=dataset_config["test_csv"],
        dataset_config=dataset_config,
        training_config=exp_config["training"],
        processors=processors,
        modalities=exp_config["modalities"],
        tokenizer=tokenizer,
        split="test"
    )

    save_dir = os.path.join(
        "qualitative",
        exp_config["dataset_name"],
        exp_config["experiment_name"]
    )
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # LOOP
    # -------------------------
    count = 0

    for batch in test_loader:

        preds, probs = get_predictions(model, batch, exp_config["modalities"])

        batch_size = len(batch["image_path"])

        for i in range(batch_size):

            image_path = batch["image_path"][i]

            gt_num = int(batch["label"][i])
            pred_num = int(preds[i].item())
            prob = float(probs[i].item())

            gt = label_map.get(gt_num, str(gt_num))
            pred = label_map.get(pred_num, str(pred_num))

            print(
                f"[DEBUG] {os.path.basename(image_path)} | "
                f"GT={gt_num}:{gt} | PRED={pred_num}:{pred}"
            )

            text = batch["raw_text"][i] if "raw_text" in batch else None

            save_sample(
                image_path=image_path,
                gt=gt,
                pred=pred,
                prob=prob,
                text=text,
                save_path=os.path.join(
                    save_dir,
                    f"{count:04d}_GT{gt}_PRED{pred}.png"
                )
            )

            count += 1

            if count >= args.num_samples:
                print(f"Saved {count} samples")
                return


if __name__ == "__main__":
    main()