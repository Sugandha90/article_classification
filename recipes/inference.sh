#!bin/bash

echo "Activating the virtual environment"
source test_env/bin/activate

echo "Starting inference..."
python src/script_inference.py

echo "Inference completed"
