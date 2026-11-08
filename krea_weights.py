"""Validate pinned Krea/Wan weights and stage them on the runtime's local disk."""
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)

# Byte sizes and LFS SHA-256 values from the two revisions in krea_assets.py.
# No duplicate root Krea checkpoint or unused Wan transformer is included.
KREA_WEIGHTS = {
    'transformer/diffusion_pytorch_model-00001-of-00003.safetensors':
        (9968228976, '5c179cb7e91005fe6e009bfb42df4ed70316f03bbc35d33e303021b33b564791'),
    'transformer/diffusion_pytorch_model-00002-of-00003.safetensors':
        (9891538864, 'a3e0f19e177dd8d83e244281ffdada88c150e080a9503dfd0f78f5acfe63563a'),
    'transformer/diffusion_pytorch_model-00003-of-00003.safetensors':
        (8717326272, '15c15fb4b8b8a181ff23b142566389344fb34973dddc8c49b4ff0dee29db2735'),
}
WAN_WEIGHTS = {
    'text_encoder/model-00001-of-00005.safetensors':
        (4972389712, 'c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99'),
    'text_encoder/model-00002-of-00005.safetensors':
        (4899225672, '481c7b2b39771c44df6dd8d13ee12ed072d731b4a650bd092885d4d52db229ad'),
    'text_encoder/model-00003-of-00005.safetensors':
        (4966309504, 'f93148bcc04052a169e1e49bfcf6125df6cf9bf243cb9c627da75266cf8e35c3'),
    'text_encoder/model-00004-of-00005.safetensors':
        (4999880704, 'a451792c739c05bca4606190cc2dd16731411bac03b4cf6aacc5767321f857c9'),
    'text_encoder/model-00005-of-00005.safetensors':
        (2885866152, '7e76e18d224531b8197a46231cb53daf7f2f6ca707130252becf933026ac4eea'),
    'vae/diffusion_pytorch_model.safetensors':
        (507591892, 'd6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793'),
}


class InvalidCheckpoint(ValueError):
    pass


def owned_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Krea staging path resolved outside its model folder')
    return path


def fingerprint(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def resolve_cached_file(path, snapshot, repo, revision):
    """Read old Drive text links only inside the pinned repo's Hub blob stores."""
    path, snapshot = Path(path), Path(snapshot).absolute()
    repository = snapshot.parent.parent
    if (snapshot.name != revision or snapshot.parent.name != 'snapshots'
            or repository.name != 'models--' + repo.replace('/', '--')):
        return path
    hub = repository.parent.resolve()
    allowed = ((repository / 'blobs').resolve(), (hub / 'blobs').resolve())
    if not all(folder.is_relative_to(hub) for folder in allowed):
        raise InvalidCheckpoint('cached blob store resolves outside its Hub cache')
    for _ in range(4):
        try:
            current = path.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise InvalidCheckpoint('cached file or blob is missing, or its link is cyclic') from exc
        if not current.is_relative_to(snapshot) and not any(current.is_relative_to(p) for p in allowed):
            raise InvalidCheckpoint('cached link resolves outside its Hub snapshot/blob stores')
        if current.stat().st_size <= 256:
            try:
                link = current.read_text(encoding='ascii').strip()
            except UnicodeError:
                return current
            if re.fullmatch(r'(?:\.\./)+blobs/(?:[0-9a-f]{2}/)?(?:[0-9a-f]{40}|[0-9a-f]{64})', link):
                candidate = current.parent / link
                if not any(candidate.resolve().is_relative_to(p) for p in allowed):
                    raise InvalidCheckpoint('cached text link resolves outside its Hub blob stores')
                logger.info('Resolving cached blob for %s', path.name)
                path = candidate
                continue
        return current
    raise InvalidCheckpoint('cached link chain exceeds the allowed depth')


def validate_metadata(path, expected):
    size, digest = expected
    if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise InvalidCheckpoint('cached metadata size/SHA-256 does not match the pinned file')
    if path.suffix == '.json':
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise InvalidCheckpoint('cached metadata is not valid JSON') from exc


def validate_checkpoint(path, expected):
    """Check the whole file's digest and parse its header, without loading tensors."""
    from safetensors import safe_open
    size, digest = expected
    try:
        actual_size = path.stat().st_size
    except FileNotFoundError as exc:
        raise InvalidCheckpoint('checkpoint is missing') from exc
    if actual_size != size:
        raise InvalidCheckpoint(f'expected {size} bytes, found {actual_size}')
    with path.open('rb') as source:
        actual_digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual_digest != digest:
        raise InvalidCheckpoint('SHA-256 does not match the pinned weight')
    try:
        with safe_open(path, framework='np') as checkpoint:
            if not checkpoint.keys():
                raise InvalidCheckpoint('checkpoint contains no tensors')
    except Exception as exc:
        raise InvalidCheckpoint(f'invalid Safetensors header: {exc}') from exc


def copy_file(source, partial, expected):
    size, digest = expected
    try:
        actual_size = source.stat().st_size
    except FileNotFoundError as exc:
        raise InvalidCheckpoint('cached checkpoint is missing') from exc
    if actual_size != size:
        raise InvalidCheckpoint(f'expected {size} bytes, found {actual_size}')
    # Hash during the copy: Drive is read once, and inference never mmaps it.
    checksum, copied, reported = hashlib.sha256(), 0, time.monotonic()
    with source.open('rb') as incoming, partial.open('wb') as outgoing:
        while chunk := incoming.read(16 * 1024 * 1024):
            outgoing.write(chunk)
            checksum.update(chunk)
            copied += len(chunk)
            if time.monotonic() - reported >= 10:
                logger.info('Staging %s: %.0f%%', source.name, 100 * copied / size)
                reported = time.monotonic()
    if copied != size or checksum.hexdigest() != digest:
        raise InvalidCheckpoint('cached checkpoint SHA-256/size does not match the pinned weight')


def copy_metadata(source, partial, expected):
    copy_file(source, partial, expected)
    if partial.suffix == '.json':
        try:
            json.loads(partial.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise InvalidCheckpoint('cached metadata is not valid JSON') from exc


def copy_checkpoint(source, partial, expected):
    copy_file(source, partial, expected)
    # Digest already checked while copying; only parse the local file's header.
    from safetensors import safe_open
    try:
        with safe_open(partial, framework='np') as checkpoint:
            if not checkpoint.keys():
                raise InvalidCheckpoint('checkpoint contains no tensors')
    except Exception as exc:
        raise InvalidCheckpoint(f'invalid Safetensors header: {exc}') from exc


def stage_snapshot(snapshot, target, repo, revision, weights, small_files, label='Krea'):
    """Reuse validated local copies; repair only invalid persistent checkpoints once."""
    from huggingface_hub import hf_hub_download
    snapshot, target = Path(snapshot), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    receipt = owned_path(target, '.validated.json')
    try:
        validated = json.loads(receipt.read_text(encoding='utf-8'))
        if not isinstance(validated, dict):
            validated = {}
    except (OSError, ValueError):
        validated = {}
    entries = {**weights, **dict(small_files)}
    pending = []
    for name, expected in entries.items():
        path = owned_path(target, name)
        try:
            record = {'sha256': expected[1], 'stat': fingerprint(path)}
            if record != validated.get(name):
                (validate_checkpoint if name in weights else validate_metadata)(path, expected)
            if path.stat().st_size != expected[0]:
                raise InvalidCheckpoint('wrong local checkpoint size')
            validated[name] = record
        except (FileNotFoundError, InvalidCheckpoint):
            pending.append(name)
    required = sum(entries[name][0] for name in pending)
    if required and shutil.disk_usage(target).free < required + 1024 ** 3:
        raise RuntimeError(f'{label} local staging needs {required / 1024**3:.1f} GiB plus 1 GiB free under '
                           f'{target}. Persistent model weights were preserved; use a VM with more disk space.')
    with tempfile.TemporaryDirectory(prefix='.stage-', dir=target) as directory:
        scratch = Path(directory).resolve()
        if scratch.parent != target:
            raise ValueError('Krea staging scratch resolved outside its model folder')
        for name in pending:
            partial = scratch / Path(name).name
            copier = copy_checkpoint if name in weights else copy_metadata
            logger.info('Copying cached %s to VM disk and validating (%.2f GiB); this is not a Hub download',
                        name, entries[name][0] / 1024**3)
            try:
                copier(resolve_cached_file(snapshot / name, snapshot, repo, revision), partial, entries[name])
            except InvalidCheckpoint as exc:
                logger.warning('Invalid cached %s: %s. Re-downloading only this file once.', name, exc)
                from colab_setup import model_cache_arguments
                repaired = hf_hub_download(repo, name, revision=revision, force_download=True,
                                           **model_cache_arguments())
                try:
                    copier(resolve_cached_file(repaired, snapshot, repo, revision), partial, entries[name])
                except InvalidCheckpoint as failure:
                    raise RuntimeError(f'{label} asset still invalid after one download: {name}. '
                                       'Check Drive storage/quota and retry; other cached models were preserved.') from failure
            destination = owned_path(target, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial.replace(destination)
            validated[name] = {'sha256': entries[name][1], 'stat': fingerprint(destination)}
            # Save per-file so a later interrupted copy can reuse completed files.
            marker = scratch / 'validated.json'
            marker.write_text(json.dumps(validated, indent=2) + '\n', encoding='utf-8')
            marker.replace(receipt)
            logger.info('Validated local %s: %s', 'checkpoint' if name in weights else 'metadata', name)
        marker = scratch / 'validated.json'
        marker.write_text(json.dumps(validated, indent=2) + '\n', encoding='utf-8')
        marker.replace(receipt)
    logger.info('%s local model files ready: %s (%d checkpoint(s) reused)',
                label, target, len(weights) - sum(name in weights for name in pending))
    return target


def local_snapshot_ready(target, files):
    """Reuse this VM's previously hash-validated copies without revisiting Drive."""
    target = Path(target).resolve()
    try:
        receipt = json.loads(owned_path(target, '.validated.json').read_text(encoding='utf-8'))
        if not isinstance(receipt, dict):
            return False
        for name, (size, digest) in files.items():
            path = owned_path(target, name)
            if path.stat().st_size != size or receipt.get(name) != {'sha256': digest, 'stat': fingerprint(path)}:
                return False
        logger.info('Reusing validated VM-local model files: %s; no download or Drive copy needed', target)
        return True
    except (OSError, ValueError, TypeError):
        return False
