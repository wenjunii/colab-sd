"""Real CPU Linear reference checks; bitsandbytes/CUDA are not installed."""
import unittest

import torch

from krea_quant import prepare_int8_linears


class LimitedLinear(torch.nn.Linear):
    def forward(self, value):
        if value.ndim > 3:
            raise RuntimeError('simulated INT8 rank limitation')
        return super().forward(value)


class QuantShapeChecks(unittest.TestCase):
    def test_rank_four_five_and_noncontiguous_inputs_match_linear_reference(self):
        torch.manual_seed(7)
        layer = LimitedLinear(8, 5)
        inputs = [torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 6, 8),
                  torch.randn(2, 4, 3, 8).transpose(1, 2)]
        self.assertEqual(prepare_int8_linears(layer, LimitedLinear), 1)
        for value in inputs:
            expected = torch.nn.functional.linear(value, layer.weight, layer.bias)
            torch.testing.assert_close(layer(value), expected)

    def test_lower_ranks_and_empty_batch_preserve_shape_and_dtype(self):
        layer = LimitedLinear(8, 5).double()
        prepare_int8_linears(layer, LimitedLinear)
        for shape in ((8,), (3, 8), (2, 3, 8), (0, 3, 4, 8)):
            value = torch.randn(shape, dtype=torch.float64)
            result = layer(value)
            self.assertEqual(result.shape, (*shape[:-1], 5))
            self.assertEqual(result.dtype, value.dtype)

    def test_idempotent_targeted_adapter_preserves_parameters_and_state_dict(self):
        model = torch.nn.ModuleDict({'quant': LimitedLinear(8, 5), 'native': torch.nn.Linear(8, 5)})
        weights = model['quant'].weight
        keys = list(model.state_dict())
        native = model['native'].forward
        self.assertEqual(prepare_int8_linears(model, LimitedLinear), 1)
        first = model['quant'].forward
        self.assertEqual(prepare_int8_linears(model, LimitedLinear), 0)
        self.assertIs(model['quant'].forward, first)
        self.assertIs(model['quant'].weight, weights)
        self.assertEqual(model['native'].forward, native)
        self.assertEqual(list(model.state_dict()), keys)
