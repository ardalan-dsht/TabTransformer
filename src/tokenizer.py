from typing import Optional, Sequence

import pandas as pd


class TabTransformerTokenizer:
    def __init__(self, tokenizer: Optional[dict | None] = None):
        self.tokenizer = tokenizer

    @staticmethod
    def extract_all_unique_tokens_in_dataset(categorical_dataset: pd.DataFrame) -> list:
        all_tokens = []
        for column in categorical_dataset.columns:
            unique_values_for_feature = categorical_dataset[column].unique().tolist()
            for value in unique_values_for_feature:
                all_tokens.append(f"{column}_{value}")
        return all_tokens

    @staticmethod
    def create_tokenizer(all_tokens: Sequence) -> dict:
        token_to_id = {}
        id_to_token = {}
        for idx, token in enumerate(all_tokens):
            token_to_id[token] = idx
            id_to_token[idx] = token
        tokenizer = {"token_to_id": token_to_id, "id_to_token": id_to_token}
        return tokenizer

    def fit(self, categorical_columns: list, dataset: pd.DataFrame) -> None:
        categorical_dataset = dataset[categorical_columns]
        all_tokens = self.extract_all_unique_tokens_in_dataset(categorical_dataset)
        self.tokenizer = self.create_tokenizer(all_tokens)
        # Categorical columns in fit and transform must be te same, but dataset can change.(train, val, test or any other thing)
        self.tokenizer["categorical_columns"] = categorical_columns

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.tokenizer == None:
            raise ValueError(
                "Tokenizer has not been initialized, either load tokenizer or create a new one with fit() method."
            )
        categorical_dataset = dataset[self.tokenizer["categorical_columns"]]
        for column in categorical_dataset:
            for value in categorical_dataset[column].unique().tolist():
                mask = categorical_dataset[column] == value
                categorical_dataset.loc[mask, column] = self.tokenizer["token_to_id"][
                    f"{column}_{value}"
                ]
        return categorical_dataset

    def fit_transform(
        self, categorical_columns: list, dataset: pd.DataFrame
    ) -> pd.DataFrame:
        self.fit(categorical_columns, dataset)
        return self.transform(dataset)

    def __len__(self) -> int:
        return len(self.tokenizer["token_to_id"])

    def load_tokenizer(self, tokenizer: dict) -> None:
        self.tokenizer = tokenizer

    def get_tokenizer(self) -> dict:
        return self.tokenizer
