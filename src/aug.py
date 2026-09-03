import yaml
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import uuid
from torchvision import transforms

from dataset.processing.text_augmentation import BackTranslationAugmenter

dataset_yaml = "src/config/datasets/crisismmd.yaml"

with open(dataset_yaml) as f:
    cfg = yaml.safe_load(f)

train_csv = cfg["train_csv"]
df = pd.read_csv(train_csv)

TEXT_COL = cfg["text_column"]
IMG_COL = cfg["image_column"]

LABEL_COL = cfg["label_column"]
LABEL_TEXT_COL = cfg["label_text_column"]

text_aug = BackTranslationAugmenter(p=1.0)

img_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
])

aug_img_dir = "data/CrisisMMD_v2.0/images_aug"
os.makedirs(aug_img_dir, exist_ok=True)


N_AUG = 1   # 1 = duplicas dataset x2, 2 = x3, etc.

augmented_rows = []


for i in tqdm(range(len(df))):

    row = df.iloc[i]

    text = str(row[TEXT_COL])
    img_path = row[IMG_COL]

    label = row[LABEL_COL]
    label_text = row[LABEL_TEXT_COL]

    augmented_rows.append(row.to_dict())

    for _ in range(N_AUG):

        try:
            aug_text = text_aug(text)
        except:
            aug_text = text

        try:
            img = Image.open(img_path).convert("RGB")
            aug_img = img_aug(img)

            new_name = f"{uuid.uuid4()}.jpg"
            new_path = os.path.join(aug_img_dir, new_name)

            aug_img.save(new_path)

        except Exception as e:
            print(f"[IMG ERROR] {img_path} -> {e}")
            new_path = img_path

        new_row = {
            TEXT_COL: aug_text,
            IMG_COL: new_path,
            LABEL_COL: label,
            LABEL_TEXT_COL: label_text
        }

        augmented_rows.append(new_row)

aug_df = pd.DataFrame(augmented_rows)

out_path = "data/CrisisMMD_v2.0/csv_splits/train_aug.csv"
aug_df.to_csv(out_path, index=False)

print("\n===== DONE =====")
print("Saved:", out_path)
print("Original:", len(df))
print("Augmented:", len(aug_df))
print("Multiplier:", len(aug_df) / len(df))