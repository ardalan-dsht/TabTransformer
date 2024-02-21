import unittest

import torch

from src.tab_transformer import TabTransformer


class TestTabTransformer(unittest.TestCase):
    """
    This is a testcase to test all functions in the TabTransformer class.
    """

    def test_forward1(self):
        categorical = torch.tensor(
            [
                [0, 3, 10, 11],
                [1, 4, 10, 12],
                [0, 4, 6, 11],
                [1, 4, 10, 11],
                [0, 3, 6, 11],
                [1, 3, 6, 11],
                [0, 3, 7, 11],
            ]
        )
        tab_transformer1 = TabTransformer(
            100, "categorical", 4, None, 4, 32, 1, 0.0, 64, 2
        )
        self.assertEqual(
            tab_transformer1(categorical).shape,
            torch.Size([7, 2]),
        )
        self.assertEqual(
            tab_transformer1(categorical).shape,
            torch.Size([7, 2]),
        )

    def test_forward2(self):
        categorical = torch.tensor(
            [
                [0, 3, 10, 11],
                [1, 4, 10, 12],
                [0, 4, 6, 11],
                [1, 4, 10, 11],
                [0, 3, 6, 11],
                [1, 3, 6, 11],
                [0, 3, 7, 11],
            ]
        )
        numerical = torch.tensor(
            [
                [22.0000, 0.0000, 7.2500],
                [38.0000, 0.0000, 71.2833],
                [26.0000, 0.0000, 7.9250],
                [35.0000, 0.0000, 53.1000],
                [35.0000, 0.0000, 8.0500],
                [54.0000, 0.0000, 51.8625],
                [2.0000, 1.0000, 21.0750],
            ]
        )
        tab_transformer2 = TabTransformer(
            token_count=250,
            count_of_input_categorical_features=16,
            count_of_input_numerical_features=3,
            n_head=4,
            dim_encoder_feedforward=64,
            num_encoder_layers=1,
            encoder_dropout=0.1,
            head_hidden_size=128,
            num_classes=3,
        )
        self.assertEqual(
            tab_transformer2(categorical, numerical).shape,
            torch.Size([7, 3]),
        )
        self.assertEqual(
            tab_transformer2(categorical, numerical).shape,
            torch.Size([7, 3]),
        )
