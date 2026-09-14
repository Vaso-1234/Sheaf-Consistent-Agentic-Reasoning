#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Rebuild aggregate.json and figures/tables from latest outputs.
python3 "$HERE/../code/analysis/aggregate.py"
python3 "$HERE/../code/analysis/make_figures.py"

# Write numeric commands from the aggregate.
python3 "$HERE/../code/analysis/write_numbers.py" > numbers.tex

# Build the PDF using tectonic (self-contained, fetches missing packages).
tectonic --keep-intermediates --keep-logs main.tex

# Sanity: no em/en dashes allowed in prose (portable across macOS BSD grep).
if python3 -c "
import sys
b = open('main.tex','rb').read()
if b'\xe2\x80\x93' in b or b'\xe2\x80\x94' in b:
    print('ERROR: em/en dash found in main.tex')
    sys.exit(2)
"; then
  :
else
  exit 2
fi

echo "PDF built: $HERE/main.pdf"
