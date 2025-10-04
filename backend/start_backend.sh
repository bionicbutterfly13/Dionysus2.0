#!/bin/bash
source flux-backend-env/bin/activate
export PYTHONPATH=/Volumes/Asylum/dev/Dionysus-2.0/backend:$PYTHONPATH
python -m src.app_factory
