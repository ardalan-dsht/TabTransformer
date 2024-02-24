from typing import Optional, Sequence

import pandas as pd


class TabTransformerTokenizer:
    """
    A class for tokenizing categorical data for TabTransformer model.
    """

    def __init__(self, tokenizer: Optional[dict | None] = None):
        self.tokenizer = tokenizer

    @staticmethod
    def extract_all_unique_tokens_in_dataset(categorical_dataset: pd.DataFrame) -> list:
        """
        Extracts all unique tokens present in the categorical dataset.

        Parameters:
        categorical_dataset (pd.DataFrame): The input categorical dataset.

        Returns:
        list: A list containing all unique tokens in the dataset.

        Example:
        all_tokens = TabTransformerTokenizer.extract_all_unique_tokens_in_dataset(categorical_data)
        """
        all_tokens = []
        for column in categorical_dataset.columns:
            unique_values_for_feature = categorical_dataset[column].unique().tolist()
            for value in unique_values_for_feature:
                all_tokens.append(f"{column}_{value}")
        return all_tokens

    @staticmethod
    def create_tokenizer(all_tokens: Sequence) -> dict:
        """
        Creates a tokenizer based on a list of all tokens.

        Parameters:
        all_tokens (Sequence): A sequence containing all tokens.

        Returns:
        dict: A dictionary with token-to-id and id-to-token mappings.

        Example:
        tokenizer = TabTransformerTokenizer.create_tokenizer(all_tokens)
        """
        token_to_id = {}
        id_to_token = {}
        for idx, token in enumerate(all_tokens):
            token_to_id[token] = idx
            id_to_token[idx] = token
        tokenizer = {"token_to_id": token_to_id, "id_to_token": id_to_token}
        return tokenizer

    def fit(self, categorical_columns: list, dataset: pd.DataFrame) -> None:
        """
        Fits the tokenizer on the dataset using the specified categorical columns.

        Parameters:
        categorical_columns (list): List of column names to tokenize.
        dataset (pd.DataFrame): The input dataset to fit the tokenizer on.

        Returns:
        None

        Example:
        tab_transformer.fit(categorical_columns, dataset)
        """
        categorical_dataset = dataset[categorical_columns]
        all_tokens = self.extract_all_unique_tokens_in_dataset(categorical_dataset)
        self.tokenizer = self.create_tokenizer(all_tokens)
        # Categorical columns in fit and transform must be te same, but dataset can change.(train, val, test or any other thing)
        self.tokenizer["categorical_columns"] = categorical_columns

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input dataset using the fitted tokenizer.

        Parameters:
        dataset (pd.DataFrame): The dataset to be transformed.

        Returns:
        pd.DataFrame: Transformed dataset with categorical columns tokenized.

        Example:
        transformed_data = tab_transformer.transform(dataset)
        """
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
        """
        Fits the tokenizer on the dataset and transforms it in a single step.

        Parameters:
        categorical_columns (list): List of column names to tokenize.
        dataset (pd.DataFrame): The input dataset to fit and transform.

        Returns:
        pd.DataFrame: Transformed dataset with categorical columns tokenized.

        Example:
        transformed_data = tab_transformer.fit_transform(categorical_columns, dataset)
        """
        self.fit(categorical_columns, dataset)
        return self.transform(dataset)

    def __len__(self) -> int:
        return len(self.tokenizer["token_to_id"])

    def load_tokenizer(self, tokenizer: dict) -> None:
        self.tokenizer = tokenizer

    def get_tokenizer(self) -> dict:
        return self.tokenizer
