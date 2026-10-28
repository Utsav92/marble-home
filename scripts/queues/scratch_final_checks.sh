#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
echo "=== pytest (full, offline)"; $PY -m pytest tests -q -p no:cacheprovider 2>&1 | tail -15
echo "=== smoke pipeline"; $PY scripts/run_pipeline.py --profile smoke --keep-going 2>&1 | grep -v "WARNING\|INFO" | tail -20
echo FINAL_CHECKS_DONE
