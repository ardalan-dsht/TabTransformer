import unittest

import torch

from src.tab_transformer import Head


class TestHead(unittest.TestCase):
    """
    This TestCase is used to test the Head in TabTransformer.
    """
    def test_forward1(self):
        x = torch.rand((7, 128))
        head1 = Head(128, 1024, 128)
        self.assertEqual(head1(x).shape, torch.Size([7, 128]))
        self.assertEqual(head1(x).shape, torch.Size([7, 128]))

    def test_forward2(self):
        x = torch.rand((32, 256))
        head2 = Head(256, 2048, 256)
        self.assertEqual(head2(x).shape, torch.Size([32, 256]))
        self.assertEqual(head2(x).shape, torch.Size([32, 256]))
