#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_ROOT"

echo "Reinstalling pyseqrna from: $REPO_ROOT"
echo "Python: $(command -v python)"
echo "Pip: $(command -v pip)"

rm -rf build dist pyseqrna.egg-info

pip uninstall -y pyseqrna || true
pip install -e .

echo "Reinstall complete."
