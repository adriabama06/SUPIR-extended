#!/bin/bash

if [ ! -f "models/.download_finished" ]; then
    mkdir -p models/checkpoints
    echo "'.download_finished' file not found. Running download_models.py..."
    python download_models.py
fi

echo "Starting..."

python gradio_demo.py --ip 0.0.0.0 --port 6688 --loading_half_params --fp8 --load_8bit_llava --use_tile_vae --outputs_folder_button
