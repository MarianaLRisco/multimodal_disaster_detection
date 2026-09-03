from importlib import import_module
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn


def compute_class_weights(csv_path, label_column, device):

    df = pd.read_csv(csv_path)

    labels = df[label_column].tolist()

    class_counts = Counter(labels)

    num_classes = len(class_counts)

    total_samples = len(labels)

    weights = []

    for class_id in range(num_classes):

        count = class_counts[class_id]

        weight = total_samples / (num_classes * count)

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float
    ).to(device)

    print("Class counts:", class_counts)
    print("Class weights:", weights)

    return weights
