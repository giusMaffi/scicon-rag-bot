#!/usr/bin/env bash

# Script per avviare l'API FastAPI in ambiente di sviluppo

set -e

# Attiva il virtualenv se esiste
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export PYTHONPATH=.

# Usa uvicorn per avviare l'app
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
