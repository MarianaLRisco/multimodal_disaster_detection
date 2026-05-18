from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def compute_metrics(
    labels,
    preds
):

    metrics = {

        "accuracy":
        accuracy_score(
            labels,
            preds
        ),

        "precision":
        precision_score(
            labels,
            preds,
            average="macro"
        ),

        "recall":
        recall_score(
            labels,
            preds,
            average="macro"
        ),

        "f1":
        f1_score(
            labels,
            preds,
            average="macro"
        )
    }

    return metrics