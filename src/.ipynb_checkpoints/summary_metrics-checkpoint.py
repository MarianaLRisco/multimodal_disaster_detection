import os
import json
from pathlib import Path


def find_metric_files(root="results"):
    metric_files = []

    for path in Path(root).rglob("test_metrics.json"):
        metric_files.append(path)

    return metric_files


def load_metrics(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_experiment_info(path):
    """
    results/Dataset/modalidad/model/exp/test_metrics.json
    """
    parts = path.parts

    try:
        dataset = parts[1]
        modality = parts[2]
        model = parts[3]
        experiment = parts[4]
    except:
        dataset = "unknown"
        modality = "unknown"
        model = "unknown"
        experiment = "unknown"

    return dataset, modality, model, experiment


def main():
    root = "results"
    metric_files = find_metric_files(root)

    summary = {
        "experiments": []
    }

    for file_path in metric_files:
        metrics = load_metrics(file_path)

        if metrics is None:
            continue

        dataset, modality, model, experiment = extract_experiment_info(file_path)

        entry = {
            "dataset": dataset,
            "modality": modality,
            "model": model,
            "experiment": experiment,
            "path": str(file_path),
            "metrics": metrics
        }

        summary["experiments"].append(entry)

    # Save summary
    output_path = Path(root) / "summary_metrics.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved summary to {output_path}")
    print(f"Total experiments: {len(summary['experiments'])}")


if __name__ == "__main__":
    main()