"""Pinned Apache-declared Krea HF runtime; never fetch the NC GitHub server."""
import hashlib
import json
from pathlib import Path

MODEL_ID = 'krea/krea-realtime-video'
MODEL_REVISION = '9a2ceea280e5d584c6a28d384a9c19e38c810f8c'
WAN_ID = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
WAN_REVISION = '38ec498cb3208fb688890f8cc7e94ede2cbd7f68'
RUNTIME_HASHES = {
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'before_denoise.py': '6831bf85ab00ac085df48c3a6ad96d61378a434c7d1f3a64fff7c0f7a8a228ff',
    'decoders.py': '9fe9395a47255b8e7d6b9101d80aa87441d61eddf2a19a8250faf325ce83c3f7',
    'denoise.py': '7c0705b6148d8455fbed1bdb428465e9d5664df26233772236d06a0ac0068c19',
    'encoders.py': '43a5cffac6820b190955d3382322a7a54c76bd0e638e61e4bebf3b23054e25ab',
    'modular_blocks.py': '2d050092ec20ed111b1aea80276238fe69593da8fd97ab7a72f8c6219ce3b1cb',
    'modular_config.json': 'e14c4a7d7645640650b1889e85ec23c551292ed506e02100d503ec3c3ca88634',
    'modular_model_index.json': 'eae35176221927b5d3c7db09b57aee1e8b2212632222b2541437ca3281719b14',
    'transformer/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'transformer/attention.py': 'e8e32379ed4569eaff374536f854677498fb69836a27d12522af10219e96c3de',
    'transformer/causal_model.py': '8e43fd4094a6af68fc5a762a0c63e8ba5154d061ea580509eedd41cecbd40e78',
    'transformer/model.py': 'e546c6cf209284c1221e1d2b77f342ea0cddee0c09ee71a58bac3b26e68e5b9f',
    'transformer/config.json': '0cd59f784f8d6ec227d4ec70f97aae9b85dfa2327c1ed89f3bf646bb8fc690ef',
    'transformer/diffusion_pytorch_model.safetensors.index.json': 'f313858edcdf6e8e5faa45d4da8c112dc75535733c16c2f64a9c698b343d1fdd',
}
WAN_PATTERNS = ('README.md', 'LICENSE*', 'text_encoder/*.json', 'text_encoder/*.safetensors', 'tokenizer/*',
                'vae/*.json', 'vae/*.safetensors', 'scheduler/*.json')

# Exact sizes/digests of the remaining files at the pinned revisions, including
# the tokenizer's two LFS files. Metadata must be checked as well as weights.
KREA_FILES = {name: (size, RUNTIME_HASHES[name]) for name, size in (
    ('__init__.py', 0), ('before_denoise.py', 47888), ('decoders.py', 5310), ('denoise.py', 10081),
    ('encoders.py', 10766), ('modular_blocks.py', 1409), ('modular_config.json', 153),
    ('modular_model_index.json', 1519), ('transformer/__init__.py', 0),
    ('transformer/attention.py', 11016), ('transformer/causal_model.py', 49526),
    ('transformer/config.json', 372), ('transformer/diffusion_pytorch_model.safetensors.index.json', 96805),
    ('transformer/model.py', 32233))}
KREA_FILES['README.md'] = (13446, '485ab341d1e594966fc25fa209c169f7d95586f3ac57ab016a51b114ec82b6ec')
WAN_FILES = {
    'README.md': (18130, 'a8371c257da24b53abca003c12ef4421fbd81eb9dfe7d3e49960016080beb51b'),
    'scheduler/scheduler_config.json': (751, '3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e'),
    'text_encoder/config.json': (854, '4087b6192155a4643f6d29fd326c4610103130674011ef4d0a53f8bce5de967d'),
    'text_encoder/model.safetensors.index.json': (22476, '8af791f24a6447aa30c95786f40adf23d3c9df97470d2f36159c8f1113f120b4'),
    'tokenizer/special_tokens_map.json': (7079, '456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270'),
    'tokenizer/spiece.model': (4548313, 'e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458'),
    'tokenizer/tokenizer.json': (16837459, '20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef'),
    'tokenizer/tokenizer_config.json': (61758, '1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd'),
    'vae/config.json': (724, 'f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527'),
}


def verify_runtime(snapshot):
    snapshot = Path(snapshot)
    for name, digest in RUNTIME_HASHES.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Krea runtime checksum mismatch: {name}; code was not loaded')


def prepare_pipeline(snapshot, wan_snapshot, root):
    """Rewrite component paths in an owned folder, preserving cached originals."""
    verify_runtime(snapshot)
    snapshot, wan_snapshot, root = Path(snapshot), Path(wan_snapshot), Path(root).resolve()
    target = (root / ('pipeline-' + MODEL_REVISION[:12])).resolve()
    if target.parent != root:
        raise ValueError('Krea pipeline folder resolved outside its runtime root')
    target.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_HASHES:
        if '/' not in name and name != 'modular_model_index.json':
            (target / name).write_bytes((snapshot / name).read_bytes())
    config = json.loads((snapshot / 'modular_model_index.json').read_text())
    # The custom transformer is loaded separately, with code verification and
    # INT8 quantization. Other components use local, pinned Wan snapshots.
    config['transformer'] = [None, None]
    for name in ('text_encoder', 'tokenizer', 'vae', 'scheduler'):
        config[name][2].update(repo=str(wan_snapshot.resolve()), revision=None)
    (target / 'modular_model_index.json').write_text(json.dumps(config, indent=2) + '\n')
    return target


def download_assets(root):
    from huggingface_hub import snapshot_download
    from krea_weights import KREA_WEIGHTS, WAN_WEIGHTS, owned_path, stage_snapshot
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                 allow_patterns=[*RUNTIME_HASHES, 'README.md', 'LICENSE*', 'transformer/*.safetensors'])
    wan = snapshot_download(WAN_ID, revision=WAN_REVISION, allow_patterns=list(WAN_PATTERNS))
    local = Path(root).resolve() / 'model-files'
    local.mkdir(parents=True, exist_ok=True)
    # Keep the persistent Hub/Drive cache intact and load from validated VM-local
    # files. Inference should not mmap multi-GB checkpoints through Drive FUSE.
    staged = stage_snapshot(snapshot, owned_path(local, 'krea-' + MODEL_REVISION),
                            MODEL_ID, MODEL_REVISION, KREA_WEIGHTS, KREA_FILES)
    staged_wan = stage_snapshot(wan, owned_path(local, 'wan-' + WAN_REVISION),
                                WAN_ID, WAN_REVISION, WAN_WEIGHTS, WAN_FILES)
    return staged, prepare_pipeline(staged, staged_wan, root)
