#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
latexmk -pdf main.tex
cp build/main.pdf knolling_bot_v2_paper.pdf
