#!/usr/bin/env bash
# Compile main.tex into main_v<n>.pdf.
#
# Usage: ./build.sh <n>
#   e.g. ./build.sh 3   -> main_v3.pdf
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <n>" >&2
    exit 1
fi

n="$1"
job="main_v${n}"

cd "$(dirname "${BASH_SOURCE[0]}")"

pdflatex -interaction=nonstopmode -jobname="$job" main.tex
bibtex "$job" || true
pdflatex -interaction=nonstopmode -jobname="$job" main.tex
pdflatex -interaction=nonstopmode -jobname="$job" main.tex

rm -f "$job.aux" "$job.log" "$job.out" "$job.blg" "$job.bbl"

echo "wrote $job.pdf"
