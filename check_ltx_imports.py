"""Check the pinned LTX source/API and public weight metadata without GPU packages."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

from ltx_assets import ensure_source, MODEL_ID, MODEL_REVISION, WEIGHTS


def main():
    with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-contract-') as directory:
        scratch = Path(directory).resolve()
        assert scratch.parent == Path(tempfile.gettempdir()).resolve()
        source = ensure_source(scratch)
        # A second pass validates the cached source rather than downloading again.
        assert ensure_source(scratch) == source
        packages = source / 'packages'
        paths = packages / 'ltx-pipelines/src/ltx_pipelines/utils/model_paths.py'
        spec = importlib.util.spec_from_file_location('ltx_model_paths_probe', paths)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            model_paths = module.ModelPaths.from_split(transformer_path='transformer', text_encoder_path='encoder',
                video_vae_path='video', audio_vae_path='audio')
            assert model_paths.transformer() == 'transformer'
            assert model_paths.embeddings_weight_paths == ('transformer', 'encoder')
        finally:
            sys.modules.pop(spec.name, None)
        checks = (
            ('ltx-pipelines/src/ltx_pipelines/distilled.py', 'DistilledPipeline', '__init__',
             {'model_paths', 'spatial_upsampler_path', 'loras', 'device', 'quantization', 'offload_mode'}),
            ('ltx-pipelines/src/ltx_pipelines/distilled.py', 'DistilledPipeline', '__call__',
             {'prompt', 'seed', 'height', 'width', 'frame_rate', 'images', 'num_frames', 'enhance_prompt'}),
            ('ltx-pipelines/src/ltx_pipelines/utils/types.py', 'ImageConditioningInput', 'at',
             {'path', 'frame_idx', 'strength'}),
            ('ltx-pipelines/src/ltx_pipelines/utils/blocks/diffusion_stage.py', 'DiffusionStage', 'from_checkpoint',
             {'checkpoint_path', 'dtype', 'device', 'quantization', 'registry', 'active_weights', 'offload_mode', 'alloc_trim_strategy'}),
            ('ltx-pipelines/src/ltx_pipelines/utils/blocks/prompt_encoder.py', 'PromptEncoder', '__init__',
             {'model_paths', 'dtype', 'device', 'registry', 'offload_mode'}),
        )
        for filename, class_name, method, required in checks:
            tree = ast.parse((packages / filename).read_text(encoding='utf-8'))
            cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
            call = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method)
            assert required <= {x.arg for x in (*call.args.args, *call.args.kwonlyargs)}
        fp8 = ast.parse((packages / 'ltx-core/src/ltx_core/quantization/fp8_cast.py').read_text())
        assert any(isinstance(x, ast.FunctionDef) and x.name == 'build_policy' for x in fp8.body)
        assert 'Apache License' in (source / 'LICENSE').read_text()
        assert 'LTX-2.x Community License Agreement' in (source / 'LICENSE-2_x').read_text()
        url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true'
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        assert metadata['sha'] == MODEL_REVISION
        entries = {x['rfilename']: x for x in metadata['siblings']}
        for name, (size, digest) in WEIGHTS.items():
            assert entries[name]['size'] == size and entries[name]['lfs']['sha256'] == digest
        print('Pinned source hash/reuse, real ModelPaths factory, pipeline/image/FP8 API signatures,')
        print('runtime/model license separation and all five weight sizes/hashes passed on CPU.')
    print('Removed only this probe\'s disposable source scratch; no model weights or GPU packages downloaded.')


if __name__ == '__main__':
    main()
