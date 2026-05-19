import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(
        self,
        feature_dim,
        attention_dim
    ):

        super().__init__()

        self.W1 = nn.Linear(
            feature_dim,
            attention_dim
        )

        self.W2 = nn.Linear(
            feature_dim,
            attention_dim
        )

        self.V = nn.Linear(
            attention_dim,
            1
        )

    def forward(
        self,
        features,
        hidden
    ):

        hidden = hidden.unsqueeze(1)

        score = torch.tanh(self.W1(features) + self.W2(hidden))

        attention_weights = F.softmax(self.V(score), dim=1)

        context_vector = (
            attention_weights * features
        )

        context_vector = torch.sum(
            context_vector,
            dim=1
        )

        return (
            context_vector,
            attention_weights
        )