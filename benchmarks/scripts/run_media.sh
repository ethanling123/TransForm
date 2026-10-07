#!/bin/bash
# Image and audio codec sweep (in-process). Needs data/m3bench/images and data/audio (benchmarks.datasets.speech).
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
mkdir -p results/media
$PY -m benchmarks.sweeps.media --workers 4 > results/logs/media.log 2>&1
echo MEDIA_DONE
