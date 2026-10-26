#!/usr/bin/env bash
# Run notebooks/p1_rag.ipynb end to end while keeping the Mac awake.
# Usage (from the project root):  bash Part1/sneha_singh/run_notebook_awake.sh
# Output: notebooks/p1_rag_executed.ipynb (the source notebook is not modified).
set -euo pipefail
cd "$(dirname "$0")/../../notebooks"
caffeinate -dimsu ../.venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('p1_rag.ipynb', as_version=4)
client = NotebookClient(nb, kernel_name='lab2-part1', timeout=-1, resources={'metadata': {'path': '.'}})
try:
    client.execute()
finally:
    nbformat.write(nb, 'p1_rag_executed.ipynb')
print('DONE')
"
