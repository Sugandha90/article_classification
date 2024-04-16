#!bin/bash

echo "Activating the virtual environment"
source test_env/bin/activate

echo "Starting inference..."
python3 src/script_inference.py.py

echo "Inference completed"
