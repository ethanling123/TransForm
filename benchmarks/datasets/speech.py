"""Synthesize a small speech set for the audio codec benchmark (macOS only: uses `say`; needs ffmpeg).

Text comes from this repo's own documents, so the passages are real prose of different lengths. Output is
16-bit mono WAV at 44.1 kHz, the usual format a speech-to-text tool would accept. Synthetic speech is cleaner
than real recordings (no background noise), so lossless codecs will do slightly better here than on field audio.

    python -m benchmarks.datasets.speech            # writes data/audio/*.wav
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Fred"]
SOURCES = ["README.md", "benchmarks/README.md", "python-server/README.md", "REPORT.md"]


def plain_passages():
    """Everyday prose (benchmarks/datasets/speech_passages.txt): the repo's own documents are full of jargon, numbers
    and symbols that a speech model transcribes differently from how they were written, which swamps a codec comparison."""
    return [line.strip() for line in (Path(__file__).with_name("speech_passages.txt")).read_text().splitlines() if line.strip()]


def passages(min_words=25, max_words=140):
    """Plain-prose paragraphs from the repo's markdown files, with code and tables removed."""
    out = []
    for name in SOURCES:
        text = Path(name).read_text()
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        for block in re.split(r"\n\s*\n", text):
            block = re.sub(r"[`*_#|>\[\]()]", " ", block)
            block = re.sub(r"https?://\S+", " ", block)
            block = re.sub(r"\s+", " ", block).strip()
            words = block.split()
            if min_words <= len(words) <= max_words and sum(c.isalpha() for c in block) > 0.7 * len(block):
                out.append(block)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/audio")
    parser.add_argument("--count", type=int, default=14)
    parser.add_argument("--source", choices=["repo", "plain"], default="repo", help="repo docs or everyday prose")
    args = parser.parse_args()

    chosen = sorted(plain_passages() if args.source == "plain" else passages(), key=len)
    step = max(1, len(chosen) // args.count)
    chosen = chosen[::step][: args.count]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(chosen):
            voice = VOICES[i % len(VOICES)]
            aiff = Path(tmp) / "x.aiff"
            # text goes through stdin: a passage starting with "-" would otherwise be read as an option
            subprocess.run(["say", "-v", voice, "-o", str(aiff), "-f", "-"], input=text.encode(), check=True)
            wav = out / f"speech_{i:02d}_{voice.lower()}.wav"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff),
                            "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(wav)], check=True)
            wav.with_suffix(".txt").write_text(text)  # the exact script, for word-error-rate checks
            print(f"{wav.name}: {len(text.split())} words, {wav.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
