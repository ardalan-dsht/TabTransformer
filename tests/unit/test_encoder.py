import unittest

import torch

from src.tab_transformer import Encoder


class TestEncoder(unittest.TestCase):
    def test1(self):
        x = torch.rand((2, 12, 12))
        encoder1 = Encoder(12, 3, 1024, 4, 0.5)
        self.assertEqual(encoder1(x).shape, torch.Size([2, 12, 12]))

    def test2(self):
        x = torch.rand((2, 14, 14))
        encoder2 = Encoder(14, 7, 256, 1, 0.0)
        self.assertEqual(encoder2(x).shape, torch.Size([2, 14, 14]))
