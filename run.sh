#!/bin/bash
cd "$(dirname "$0")"
# Show output here AND save it to miniflow.log
.venv/bin/python -u miniflow.py "$@" 2>&1 | tee miniflow.log
