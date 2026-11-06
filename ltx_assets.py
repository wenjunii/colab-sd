"""Pinned Apache runtime source and gated LTX-2.5 split weights."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile

CODE_REVISION = '4f521619ae0029e8fbb64d555030c2b121a33b4f'
CODE_SHA256 = '1495a71b05c8ef86bb21da2238915e1999de85b0a8cab80a5c094731be6c8a02'
MODEL_ID = 'Lightricks/LTX-2.5'
MODEL_REVISION = '49b093a39c19169a8a81721dcc541e07e945232f'
TRANSFORMER = 'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors'
TEXT_ENCODER = 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
VIDEO_VAE = 'vae/ltx-2.5-video-vae-conv-bf16.safetensors'
AUDIO_VAE = 'vae/ltx-2.5-audio-vae-bf16.safetensors'
UPSCALER = 'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'
WEIGHTS = {
    TRANSFORMER: (42018190584, '31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4'),
    TEXT_ENCODER: (26263858182, 'ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1'),
    VIDEO_VAE: (1452269922, '685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d'),
    AUDIO_VAE: (364866540, 'c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5'),
    UPSCALER: (995778752, 'eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8'),
}


def unpack_source(payload, target):
    if hashlib.sha256(payload).hexdigest() != CODE_SHA256:
        raise ValueError('LTX source archive SHA-256 does not match the pinned revision')
    target = Path(target).resolve()
    prefix = f'LTX-2-{CODE_REVISION}'
    files = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 5000 or sum(x.file_size for x in archive.infolist()) > 100 * 1024**2:
            raise ValueError('LTX source archive exceeds the extraction limit')
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != prefix or '..' in parts or '\\' in member.filename:
                raise ValueError('LTX source archive has an invalid path')
            if member.is_dir():
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('LTX source archive contains a symbolic link')
            relative = Path(*parts[1:])
            destination = (target / relative).resolve()
            if destination == target or not destination.is_relative_to(target):
                raise ValueError('LTX extraction escaped its source directory')
            data = archive.read(member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            files[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    for name in ('LICENSE', 'LICENSE-2_x', 'packages/ltx-core/pyproject.toml',
                 'packages/ltx-pipelines/pyproject.toml'):
        if name not in files:
            raise ValueError('LTX source archive is missing a required file: ' + name)
    (target / '.bridge-source.json').write_text(json.dumps(files, indent=2), encoding='utf-8')


def ensure_source(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / ('ltx-source-' + CODE_REVISION)
    if target.exists():
        receipt = target / '.bridge-source.json'
        if not receipt.is_file():
            raise RuntimeError('Existing LTX source folder is unmanaged; use a different runtime folder')
        files = json.loads(receipt.read_text(encoding='utf-8'))
        for name, digest in files.items():
            path = (target / name).resolve()
            if not path.is_relative_to(target) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Pinned LTX source changed; preserve it and use a new runtime folder')
        return target
    url = f'https://codeload.github.com/Lightricks/LTX-2/zip/{CODE_REVISION}'
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = response.read(10 * 1024**2 + 1)
    with tempfile.TemporaryDirectory(prefix='.ltx-source-', dir=root) as directory:
        scratch = Path(directory).resolve()
        if scratch.parent != root:
            raise ValueError('LTX source scratch escaped the runtime folder')
        unpack_source(payload, scratch)
        scratch.replace(target)
    return target


def download_models(root):
    import shutil
    from huggingface_hub import hf_hub_download, snapshot_download
    from krea_weights import stage_snapshot
    target = Path(root).resolve() / 'model-files' / MODEL_REVISION
    target.mkdir(parents=True, exist_ok=True)
    missing = sum(size for name, (size, _) in WEIGHTS.items() if not (target / name).is_file())
    if shutil.disk_usage(target).free < missing + 1024**3:
        raise RuntimeError(f'LTX local staging needs {missing / 1024**3:.1f} GiB plus 1 GiB free')
    # Check access with metadata only, before requesting any of the large weights.
    try:
        hf_hub_download(MODEL_ID, AUDIO_VAE, revision=MODEL_REVISION, dry_run=True)
    except Exception as exc:
        raise RuntimeError('LTX model access failed. Accept the model terms on Hugging Face and '
                           'provide HF_TOKEN through Colab Secrets; do not paste it into chat or source.') from exc
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(WEIGHTS))
    return stage_snapshot(snapshot, target, MODEL_ID, MODEL_REVISION, WEIGHTS, {}, label='LTX-2.5')
