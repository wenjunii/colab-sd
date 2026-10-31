from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import tensorrt_backend as backend


class TensorRTChecks(unittest.TestCase):
    def test_package_matches_current_torch_without_replacing_cuda_build(self):
        self.assertEqual(backend.package_requirement('2.11.0+cu130', (3, 13)), 'torch-tensorrt==2.11.*')
        for torch_version, python in [('2.5.1', (3, 13)), ('2.11.0', (3, 10))]:
            with self.assertRaises(RuntimeError):
                backend.package_requirement(torch_version, python)

    def test_cache_separates_modes_resolutions_and_gpu_software_versions(self):
        args = ('engines', 512, 384, 1, '2.11.0+cu130', '2.11.0', (8, 0))
        path = backend.cache_directory(*args)
        self.assertNotEqual(path, backend.cache_directory(*args[:3], 2, *args[4:]))
        self.assertIn('512x384', str(path))
        self.assertIn('sm80', str(path))

    def test_adapter_preserves_streamdiffusion_tuple_and_timestep_contract(self):
        compiled = Mock(return_value=torch.ones(1, 4, 2, 2))
        source = SimpleNamespace(config=SimpleNamespace(), dtype=torch.float16, device=torch.device('cpu'))
        adapter = backend.TensorRTUNet(compiled, source, 1)
        latent = torch.zeros(1, 4, 2, 2)
        result = adapter(latent, torch.tensor([999]), torch.zeros(1, 77, 8), return_dict=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(compiled.call_args.args[1].dtype, torch.float32)
        with self.assertRaises(ValueError):
            adapter(torch.zeros(2, 4, 2, 2), torch.tensor([999, 500]), torch.zeros(2, 77, 8))

    def test_compile_requires_a_real_engine_and_validates_outputs(self):
        class FakeUNet(torch.nn.Module):
            config = SimpleNamespace(in_channels=4, cross_attention_dim=8)
            device = torch.device('cpu')
            dtype = torch.float32

            def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
                return (sample * 0.5,)

        class TorchTensorRTModule(torch.nn.Module):
            def __init__(self, graph):
                super().__init__()
                self.graph = graph

            def forward(self, *args):
                return self.graph(*args)

        compiler = Mock(side_effect=lambda graph, **kwargs: TorchTensorRTModule(graph))
        module = SimpleNamespace(__version__='2.11.0', compile=compiler)
        with tempfile.TemporaryDirectory(prefix='colab-sd-trt-test-') as name:
            root = Path(name).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            with patch.dict('sys.modules', {'torch_tensorrt': module}), \
                    patch.object(backend, 'package_requirement'), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'get_device_capability', return_value=(8, 0)), \
                    patch.object(torch.cuda, 'synchronize'), patch('builtins.print'):
                adapter = backend.compile_unet(FakeUNet(), 32, 16, 2, root)
                self.assertEqual(adapter.batch_size, 2)
                self.assertTrue(compiler.call_args.kwargs['require_full_compilation'])
                self.assertTrue(compiler.call_args.kwargs['use_explicit_typing'])
                self.assertNotIn('enabled_precisions', compiler.call_args.kwargs)
                self.assertEqual(compiler.call_args.kwargs['inputs'][0].shape, (2, 4, 2, 4))
                compiler.side_effect = lambda graph, **kwargs: graph
                with self.assertRaisesRegex(RuntimeError, 'no TensorRT engines'):
                    backend.compile_unet(FakeUNet(), 32, 16, 1, root)
