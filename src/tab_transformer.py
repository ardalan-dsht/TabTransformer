from typing import Optional

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        count_of_input_features: int,
        n_encoder_head: int,
        dim_encoder_feedforward: int,
        num_encoder_layers: int,
        encoder_dropout: float,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            count_of_input_features,
            n_encoder_head,
            dim_encoder_feedforward,
            encoder_dropout,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_stack(x)
        return x


class Head(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class TabTransformer(nn.Module):
    def __init__(
        self,
        token_count: int,
        model_type: str = "categorical_numerical",
        count_of_input_categorical_features: Optional[int] = None,
        count_of_input_numerical_features: Optional[int] = None,
        n_head: int = 4,
        dim_encoder_feedforward: int = 64,
        num_encoder_layers: int = 1,
        encoder_dropout: float = 0.1,
        head_hidden_size: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.column_embedder = nn.Embedding(
            token_count, count_of_input_categorical_features
        )
        self.encoder_stack = Encoder(
            count_of_input_categorical_features,
            n_head,
            dim_encoder_feedforward,
            num_encoder_layers,
            encoder_dropout,
        )
        if count_of_input_numerical_features is not None:
            self.layer_norm_for_numerical = nn.LayerNorm(
                count_of_input_numerical_features
            )
            count_of_features = (
                count_of_input_numerical_features + count_of_input_categorical_features
            )
        else:
            self.model_type = "categorical"
            count_of_features = count_of_input_categorical_features
        self.head = Head(count_of_features, head_hidden_size, num_classes)

    def forward(
        self,
        categorical: torch.Tensor,
        numerical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        categorical = self.column_embedder(categorical)
        categorical = self.encoder_stack(categorical)
        head_input = categorical.mean(dim=1)
        if self.model_type == "categorical_numerical":
            numerical = self.layer_norm_for_numerical(numerical)
            head_input = torch.cat([head_input, numerical], dim=1)
        output = self.head(head_input)
        return output
