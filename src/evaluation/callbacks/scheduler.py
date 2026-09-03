from transformers import (
    get_cosine_schedule_with_warmup
)

def build_scheduler(
    optimizer,
    scheduler_config,
    training_config,
    train_dataloader
):

    name = scheduler_config["name"]
    epochs = training_config["epochs"]
    total_steps = (
        len(train_dataloader) * epochs
    )

    warmup_steps = int(
        total_steps * scheduler_config.get(
            "warmup_ratio",
            0.1
        )
    )

    if name == "cosine_warmup":

        scheduler = (
            get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        )

        return scheduler

    return None