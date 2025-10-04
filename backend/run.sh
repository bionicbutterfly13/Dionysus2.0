#!/bin/bash
source flux-backend-env/bin/activate
cd src
python -m uvicorn app_factory:app --host 127.0.0.1 --port 9127 --reload
