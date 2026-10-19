#!/bin/bash
# Lossy-arm experiments with local models (no API key): clean codec timings, document OCR with ground truth,
# audio under noise, and natural images with a rerun control. Everything resumes if interrupted.
# Needs ollama serving qwen2.5vl:7b and mlx-whisper (see benchmarks/README.md).
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
mkdir -p results/lossy results/logs

$PY -m benchmarks.sweeps.media --workers 1 > results/logs/lossy_media.log 2>&1
$PY -m benchmarks.sweeps.lossy documents > results/logs/lossy_documents.log 2>&1
for snr in 20 10 5; do
  $PY -m benchmarks.sweeps.lossy audio --snr $snr > results/logs/lossy_audio_snr$snr.log 2>&1
done
$PY -m benchmarks.sweeps.lossy images --stride 3 --limit 80 \
  --arms jpeg-q95,png-rerun,jpeg-q75,webp-q80,webp-q60,avif-q60,avif-q40,png-1024px > results/logs/lossy_images.log 2>&1
echo LOSSY_DONE
