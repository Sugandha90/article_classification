#!bin/bash

echo "Activating the virtual environment"
source test_env/bin/activate

echo "Starting training..."
python3 src/script_train.py

echo "Training completed"