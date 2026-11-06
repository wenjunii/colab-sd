"""Real CPU Linear reference checks; bitsandbytes/CUDA are not installed."""
import unittest
import logging
from unittest.mock import patch

import torch

from krea_quant import (prepare_int8_linears, validate_int8_profile, INT8_THRESHOLDS,
                        int8_engine_label, limit_cast_warnings, _CastWarningOnce, configure_allocator,
                        prepare_vae_layout)


class LimitedLinear(torch.nn.Linear):
    def forward(self, value):
        if value.ndim > 3:
            raise RuntimeError('simulated INT8 rank limitation')
        return super().forward(value)


class QuantShapeChecks(unittest.TestCase):
    def test_fast_vae_layout_preserves_convolution_values_and_other_parameters(self):
        torch.manual_seed(7)
        vae = torch.nn.ModuleDict({'conv': torch.nn.Conv3d(2, 4, 3, padding=1),
                                   'linear': torch.nn.Linear(4, 3)})
        value = torch.randn(1, 2, 3, 8, 8)
        reference = vae['conv'](value)
        weights = {key: tensor.clone() for key, tensor in vae.state_dict().items()}
        linear_parameter = vae['linear'].weight
        prepare_vae_layout(vae, 'balanced')
        self.assertTrue(vae['conv'].weight.is_contiguous())
        prepare_vae_layout(vae, 'fast')
        self.assertTrue(vae['conv'].weight.is_contiguous(memory_format=torch.channels_last_3d))
        torch.testing.assert_close(vae['conv'](value), reference)
        self.assertIs(vae['linear'].weight, linear_parameter)
        self.assertEqual(set(vae.state_dict()), set(weights))
        for key, tensor in vae.state_dict().items():
            torch.testing.assert_close(tensor, weights[key], rtol=0, atol=0)

    def test_allocator_default_respects_both_user_override_aliases(self):
        environment = {'OTHER': 'kept'}
        self.assertIs(configure_allocator(environment), environment)
        self.assertEqual(environment, {'OTHER': 'kept', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True'})
        for name in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
            configured = {name: 'backend:cudaMallocAsync', 'OTHER': 'kept'}
            self.assertEqual(configure_allocator(configured), {name: 'backend:cudaMallocAsync', 'OTHER': 'kept'})

    def test_profiles_reject_unknown_settings_and_explain_actual_engine(self):
        for value in (None, False, 0, [], 'turbo', 'FAST'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_int8_profile(value)
        self.assertEqual(INT8_THRESHOLDS[validate_int8_profile('fast')], 0.0)
        self.assertEqual(INT8_THRESHOLDS[validate_int8_profile('balanced')], 6.0)
        self.assertIn('Fast INT8', int8_engine_label('fast'))
        self.assertIn('Balanced INT8', int8_engine_label('balanced'))

    def test_only_duplicate_pinned_cast_warnings_are_suppressed(self):
        once = _CastWarningOnce()
        template = 'MatMul8bitLt: inputs will be cast from %s to float16 during quantization'
        def record(message=template, args=('torch.bfloat16',), level=logging.WARNING):
            return logging.LogRecord('bnb', level, '', 0, message, args, None)
        self.assertTrue(once.filter(record()))
        self.assertFalse(once.filter(record()))
        self.assertTrue(once.filter(record(args=('torch.float32',))))
        self.assertTrue(once.filter(record(level=logging.ERROR)))
        self.assertTrue(once.filter(record('Important CUDA warning', ())))
        self.assertTrue(once.filter(record('Important CUDA warning', ())))
        target = logging.getLogger('bitsandbytes.autograd._functions')
        with patch.object(target, 'filters', []):
            limit_cast_warnings()
            limit_cast_warnings()
            self.assertEqual(len(target.filters), 1)

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
