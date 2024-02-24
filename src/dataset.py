from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabTransformerDataset(Dataset):
    def __init__(
        self,
        labels: pd.Series,
        categorical_dataset: pd.DataFrame,
        numerical_dataset: Optional[pd.DataFrame] = None,
    ):
        self.categorical_dataset = categorical_dataset
        if numerical_dataset is not None:
            self.model_type = "categorical_numerical"
            self.numerical_dataset = numerical_dataset
        else:
            self.model_type = "categorical"
        self.labels = labels

    def __len__(self) -> int:
        return len(self.categorical_dataset)

    def __getitem__(self, index: int) -> tuple:
        categorical_sample = self.categorical_dataset.iloc[index].values.astype(
            np.int32
        )
        categorical_sample = torch.from_numpy(categorical_sample)
        label_sample = self.labels.iloc[index]
        if self.model_type == "categorical_numerical":
            numerical_sample = self.numerical_dataset.iloc[index].values.astype(
                np.float32
            )
            numerical_sample = torch.from_numpy(numerical_sample)
            numerical_sample = numerical_sample.to(torch.long)
            return categorical_sample, label_sample, numerical_sample
        else:
            return categorical_sample, label_sample
