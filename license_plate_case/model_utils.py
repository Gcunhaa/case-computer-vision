# model_utils.py
import os
import requests
from pathlib import Path
import hashlib
import sys
from tqdm import tqdm

# Default model URL 
MODEL_URL = "https://github.com/Gcunhaa/case-computer-vision/raw/refs/heads/main/license_plate_case/models/best.pt"

def get_model_path() -> Path:
    package_dir = Path(__file__).parent.absolute()
    models_dir = package_dir / "models"
    model_path = models_dir / "best.pt"
    
    if not model_path.exists():
        download_model(model_path)
    
    return model_path

def download_model(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        # Get total file size from headers
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"Downloading license plate model")
        with open(model_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading model"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        print(f"Model downloaded to {model_path}")
    except Exception as e:
        print(f"Download failed: {str(e)}")
        sys.exit(1)