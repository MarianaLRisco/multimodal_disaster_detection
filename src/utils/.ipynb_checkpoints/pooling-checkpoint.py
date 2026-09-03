import torch


def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output.last_hidden_state
    )

    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded,
        1
    ) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


def cls_pooling(model_output):

    return model_output.last_hidden_state[:, 0]


def last_token_pooling(
    model_output,
    attention_mask
):

    sequence_lengths = (
        attention_mask.sum(dim=1) - 1
    )

    batch_size = (
        model_output.last_hidden_state.shape[0]
    )

    return model_output.last_hidden_state[
        torch.arange(batch_size),
        sequence_lengths
    ]