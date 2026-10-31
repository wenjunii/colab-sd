"""Shared presets, control validation, and bounded input frame reassembly."""
import math
import time

PROTOCOL_VERSION = 1
UPLOAD_CHUNK_BYTES = 4096
OUTPUT_CHUNK_BYTES = 2048
MAX_INPUT_FRAME_BYTES = 512 * 1024
OUTPUT_PRESETS = (
    ('256', '256 x 256 (1:1 Fast)', 256, 256),
    ('384', '384 x 384 (1:1)', 384, 384),
    ('512', '512 x 512 (1:1 Default)', 512, 512),
    ('768', '768 x 768 (1:1)', 768, 768),
    ('512x384', '512 x 384 (4:3)', 512, 384),
    ('768x576', '768 x 576 (4:3)', 768, 576),
    ('1024x768', '1024 x 768 (4:3 Max)', 1024, 768),
    ('384x512', '384 x 512 (3:4)', 384, 512),
    ('576x768', '576 x 768 (3:4)', 576, 768),
    ('768x1024', '768 x 1024 (3:4 Max)', 768, 1024),
    ('512x288', '512 x 288 (16:9)', 512, 288),
    ('768x432', '768 x 432 (16:9)', 768, 432),
    ('1024x576', '1024 x 576 (16:9)', 1024, 576),
    ('288x512', '288 x 512 (9:16)', 288, 512),
    ('432x768', '432 x 768 (9:16)', 432, 768),
    ('576x1024', '576 x 1024 (9:16)', 576, 1024),
)
OUTPUT_RESOLUTIONS = tuple((width, height) for _, _, width, height in OUTPUT_PRESETS)


def integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if isinstance(value, float) and value != number:
        raise ValueError(f'{name} must be an integer')
    return number


def resolution(data, current=(512, 512)):
    size = (integer(data.get('width', current[0]), 'width'),
            integer(data.get('height', current[1]), 'height'))
    if size not in OUTPUT_RESOLUTIONS:
        raise ValueError('Choose a supported output preset, up to 1024 x 768 or 768 x 1024')
    return size


def validate_controls(data, current=(512, 512)):
    """Validate the whole patch before mutating any model or application state."""
    if not isinstance(data, dict):
        raise ValueError('Control message must be a JSON object')
    clean = {}
    if 'width' in data or 'height' in data:
        clean['width'], clean['height'] = resolution(data, current)
    for name, lower, upper in (('strength', 0.1, 1.0), ('blend', 0.0, 1.0)):
        if name in data:
            try:
                value = float(data[name])
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f'{name} must be a finite number') from exc
            if isinstance(data[name], bool) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name} must be between {lower} and {upper}')
            clean[name] = value
    if 'seed' in data:
        seed = integer(data['seed'], 'seed')
        if not 0 <= seed <= 4294967295:
            raise ValueError('seed must be between 0 and 4294967295')
        clean['seed'] = seed
    if 'mode' in data:
        mode = str(data['mode']).lower()
        aliases = {'txt2img': 'txt2img', 'text': 'txt2img', 'img2img': 'img2img', 'v2v': 'img2img'}
        if mode not in aliases:
            raise ValueError('mode must be txt2img or img2img')
        clean['mode'] = aliases[mode]
    if 'prompt' in data and 'prompt_a' not in data:
        data = {**data, 'prompt_a': data['prompt']}
    for name in ('prompt_a', 'prompt_b', 'neg_prompt'):
        if name in data:
            value = data[name]
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f'{name} must be text of at most 4096 characters')
            if name == 'prompt_a' and not value.strip():
                raise ValueError('Prompt A cannot be empty')
            clean[name] = value.strip()
    return clean


class FrameAssembler:
    """Keep one incomplete frame, discarding stale or inconsistent chunks."""
    def __init__(self, max_bytes=MAX_INPUT_FRAME_BYTES, timeout=1.0):
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.completed_id = None
        self.clear()

    def clear(self):
        self.frame_id = None
        self.chunks = []
        self.size = 0
        self.started = 0.0

    def feed(self, raw, now=None):
        now = time.monotonic() if now is None else now
        if not raw.startswith(b'FRM:'):
            if not raw.startswith(b'\xff\xd8') or len(raw) > self.max_bytes:
                raise ValueError('Expected a bounded JPEG input frame')
            self.clear()
            return raw
        if len(raw) < 8 or len(raw) > UPLOAD_CHUNK_BYTES + 7:
            self.clear()
            raise ValueError('Invalid input chunk size')
        frame_id, index, count = raw[4:7]
        if not count or index >= count:
            self.clear()
            raise ValueError('Invalid input chunk index or count')
        if frame_id == self.completed_id:
            return None
        if self.frame_id != frame_id or now - self.started > self.timeout:
            self.clear()
            self.frame_id = frame_id
            self.chunks = [None] * count
            self.started = now
        if len(self.chunks) != count:
            self.clear()
            raise ValueError('Input chunk count changed within a frame')
        previous = self.chunks[index]
        self.size += len(raw) - 7 - (len(previous) if previous is not None else 0)
        if self.size > self.max_bytes:
            self.clear()
            raise ValueError('Input frame exceeds the byte limit')
        self.chunks[index] = raw[7:]
        if all(chunk is not None for chunk in self.chunks):
            result = b''.join(self.chunks)
            self.completed_id = frame_id
            self.clear()
            return result
        return None
