import os
from pathlib import Path
import re
import random
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PYTHONPATH"))
DATASET_ROOT_DIR = PROJECT_ROOT / 'datasets' / 'card_recognizer'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def find_image_path(card_id, root_dir = DATASET_ROOT_DIR / 'cards'):
    root_path = Path(root_dir)
    for image_path in root_path.rglob(f"{card_id}"):
        return image_path
    return None

def extract_prefix(card_id):
    match = re.match(r'^[^-]+', card_id)
    return match.group(0) if match else ''