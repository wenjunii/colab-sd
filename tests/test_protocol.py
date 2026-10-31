import unittest

from bridge_protocol import FrameAssembler, OUTPUT_RESOLUTIONS, resolution, validate_controls


def packet(frame, index, count, payload=b'x'):
    return b'FRM:' + bytes([frame, index, count]) + payload


class ProtocolChecks(unittest.TestCase):
    def test_controls_normalize_legacy_names_and_numeric_strings(self):
        self.assertEqual(validate_controls({'prompt': ' hello ', 'seed': '42', 'mode': 'v2v'}),
                         {'prompt_a': 'hello', 'seed': 42, 'mode': 'img2img'})

    def test_controls_reject_nonfinite_fractional_and_out_of_range_values(self):
        for data in ({'blend': float('nan')}, {'strength': float('inf')}, {'strength': False},
                     {'strength': -1}, {'seed': -1}, {'seed': 2.3}, {'seed': 2**32},
                     {'mode': 'unknown'}, {'prompt_a': ''}, {'prompt_b': None}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_controls(data)

    def test_resolution_presets_and_integer_validation(self):
        for width, height in OUTPUT_RESOLUTIONS:
            self.assertEqual(resolution({'width': width, 'height': height}), (width, height))
        for width in (True, 512.1, float('nan'), 'nope'):
            with self.subTest(width=width), self.assertRaises(ValueError):
                resolution({'width': width, 'height': 512})

    def test_out_of_order_chunks_and_duplicate_chunks(self):
        assembler = FrameAssembler(max_bytes=6)
        self.assertIsNone(assembler.feed(packet(1, 1, 2, b'def')))
        self.assertIsNone(assembler.feed(packet(1, 1, 2, b'def')))
        self.assertEqual(assembler.feed(packet(1, 0, 2, b'abc')), b'abcdef')
        self.assertIsNone(assembler.feed(packet(1, 0, 2, b'abc')))

    def test_incomplete_frame_is_replaced_and_frame_id_wraps(self):
        assembler = FrameAssembler()
        assembler.feed(packet(254, 0, 2))
        self.assertEqual(assembler.feed(packet(255, 0, 1, b'new')), b'new')
        self.assertEqual(assembler.feed(packet(0, 0, 1, b'wrapped')), b'wrapped')

    def test_expired_chunks_are_not_mixed_with_a_new_attempt(self):
        assembler = FrameAssembler(timeout=1)
        assembler.feed(packet(1, 0, 2, b'old'), now=0)
        self.assertIsNone(assembler.feed(packet(1, 1, 2, b'new-b'), now=2))
        self.assertEqual(assembler.feed(packet(1, 0, 2, b'new-a'), now=2.1), b'new-anew-b')

    def test_bad_headers_and_size_limits_clear_partial_buffers(self):
        for raw in (b'FRM:', packet(1, 2, 2), packet(1, 0, 0), packet(1, 0, 1, b'x' * 4097)):
            with self.subTest(raw_length=len(raw)), self.assertRaises(ValueError):
                FrameAssembler().feed(raw)
        assembler = FrameAssembler(max_bytes=4)
        assembler.feed(packet(1, 0, 2, b'abc'))
        with self.assertRaises(ValueError):
            assembler.feed(packet(1, 1, 2, b'def'))
        self.assertEqual(assembler.chunks, [])

    def test_chunk_count_changes_and_invalid_raw_frames_are_rejected(self):
        assembler = FrameAssembler()
        assembler.feed(packet(1, 0, 2))
        with self.assertRaises(ValueError):
            assembler.feed(packet(1, 1, 3))
        with self.assertRaises(ValueError):
            assembler.feed(b'not a jpeg')
        self.assertEqual(assembler.feed(b'\xff\xd8jpeg'), b'\xff\xd8jpeg')
