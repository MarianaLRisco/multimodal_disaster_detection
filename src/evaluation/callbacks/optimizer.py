from torch.optim import AdamW
from torch.optim import SGD


def build_optimizer(
    model,
    optimizer_config,
    training_config
):

    name = optimizer_config["name"].lower()

    lr = float(
        training_config["learning_rate"]
    )

    weight_decay = float(
        training_config["weight_decay"]
    )

    if name == "adamw":

        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "sgd":

        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    else:

        raise ValueError(
            f"Unknown optimizer: {name}"
        )