"""The opt-in model-aware arms: image scaled to a vision model's limits, speech as 16 kHz mono.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

from PIL import Image  # noqa: E402

import arms  # noqa: E402
import shared  # noqa: E402
import server_app as S  # noqa: E402


def png(width: int, height: int) -> bytes:
    """Photo-like (incompressible) pixels, so scaling down makes the file smaller."""
    buf = io.BytesIO()
    Image.frombytes("RGB", (width, height), os.urandom(width * height * 3)).save(buf, "PNG")
    return buf.getvalue()


def sparse_png(width: int, height: int) -> bytes:
    """A dot every 7 px on white: tiny as a PNG, but resampling blurs it into something larger."""
    im = Image.new("RGB", (width, height), (255, 255, 255))
    px = im.load()
    for x in range(0, width, 7):
        for y in range(0, height, 7):
            px[x, y] = ((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def wav(seconds: float = 1.0, rate: int = 44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(rate * seconds))))
    return buf.getvalue()


class ImageFit(unittest.TestCase):
    def test_graphics_that_compress_better_at_full_size_are_left_alone(self):
        original = sparse_png(3000, 2000)
        self.assertEqual(arms.fit_image_to_model(original), original)

    def test_a_large_image_is_scaled_to_the_model_limits(self):
        out = Image.open(io.BytesIO(arms.fit_image_to_model(png(2400, 1600))))
        self.assertLessEqual(max(out.size), arms.VISION_MAX_EDGE)
        self.assertLessEqual(math.ceil(out.size[0] / 28) * math.ceil(out.size[1] / 28), arms.VISION_MAX_TOKENS)
        self.assertAlmostEqual(out.size[0] / out.size[1], 1.5, delta=0.02)

    def test_an_image_that_already_fits_is_returned_untouched(self):
        original = png(800, 600)
        self.assertEqual(arms.fit_image_to_model(original), original)

    def test_a_square_image_respects_the_token_limit_not_just_the_edge(self):
        out = Image.open(io.BytesIO(arms.fit_image_to_model(png(1568, 1568))))
        self.assertLessEqual(math.ceil(out.size[0] / 28) * math.ceil(out.size[1] / 28), arms.VISION_MAX_TOKENS)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class Audio(unittest.TestCase):
    def test_speech_becomes_16k_mono_wav(self):
        out = wave.open(io.BytesIO(arms.audio_to_16k_mono(wav())))
        self.assertEqual((out.getframerate(), out.getnchannels(), out.getsampwidth()), (16000, 1, 2))
        self.assertAlmostEqual(out.getnframes() / 16000, 1.0, delta=0.05)


class ThroughTheServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)

    def register(self, data: bytes, mime: str) -> str:
        return self.client.post("/materialized-raw", content=data, headers={"Content-Type": mime}).json()["result_id"]

    def test_the_arms_are_registered_but_never_selectable(self):
        for name in ("image_fit_blob", "audio_16k_blob"):
            self.assertFalse(arms.REGISTRY.get(name).selectable)
            self.assertFalse(arms.REGISTRY.get(name).lossless)

    def test_vision_consumer_gets_the_scaled_image(self):
        original = png(2400, 1600)
        rid = self.register(original, "image/png")
        out = S.large_result_auto(0, 0, result_id=rid, consumer="vision").structured_content
        self.assertEqual(out["chosen_format"], "image_fit_blob")
        self.assertFalse(out["decode"]["lossless"])
        body = self.client.get(out["decode"]["url"].replace("http://localhost:8000", "")).content
        self.assertLess(len(body), len(original))
        self.assertLessEqual(max(Image.open(io.BytesIO(body)).size), arms.VISION_MAX_EDGE)

    def test_without_a_consumer_the_original_is_what_the_selector_picks_from(self):
        rid = self.register(png(2400, 1600), "image/png")
        self.assertNotEqual(S.large_result_auto(0, 0, result_id=rid).structured_content["chosen_format"], "image_fit_blob")

    def test_a_consumer_that_does_not_match_the_payload_falls_through(self):
        rid = self.register(b"plain text " * 500, "text/plain")
        self.assertNotEqual(S.large_result_auto(0, 0, result_id=rid, consumer="vision").structured_content["chosen_format"], "image_fit_blob")

    def test_unknown_consumer_is_an_error(self):
        rid = self.register(png(100, 100), "image/png")
        with self.assertRaises(ValueError):
            S.large_result_auto(0, 0, result_id=rid, consumer="telepathy")

    def test_fetch_arm_refuses_the_wrong_mime(self):
        rid = self.register(b"abc" * 100, "text/plain")
        with self.assertRaises(ValueError):
            S.fetch_arm("image_fit_blob", rid)


if __name__ == "__main__":
    unittest.main()
