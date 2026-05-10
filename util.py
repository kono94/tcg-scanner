import os
import cv2
from pathlib import Path
import re
import random
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PYTHONPATH", Path.cwd()))
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

def display_top_k_images(labels, distances, top_k):
    images = []
    max_height = 0
    for i, card_id in enumerate(labels):
        image_path = find_image_path(card_id, DATASET_ROOT_DIR / 'cards')
        if image_path:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image for card ID: {card_id} at {image_path}")
                continue
            max_height = max(max_height, image.shape[0])
            images.append((image, card_id, distances[i]))
        else:
            print(f"No image found for card ID: {card_id}")
    
    if not images:
        print("No valid images to display.")
        return
    
    # Pad images to max_height and prepare text annotations
    padded_images = []
    for image, card_id, dist in images:
        h, w = image.shape[:2]
        top_pad = (max_height - h) // 2
        bottom_pad = max_height - h - top_pad
        padded_image = cv2.copyMakeBorder(
            image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # Add text annotation with red text and black background
        text = f"{card_id} (Dist: {dist:.4f})"
        font_scale = 1
        thickness = 1
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_w, text_h = text_size
        text_w = min(text_w + 10, w)
        text_bg = np.zeros((text_h + 10, text_w, 3), dtype=np.uint8)
        cv2.putText(
            text_bg, text, (5, text_h + 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA
        )
        padded_image[0:text_h + 10, 0:text_w] = text_bg
        padded_images.append(padded_image)
    
    # Combine images horizontally
    combined_image = np.hstack(padded_images)
    window_name = f"Top-{top_k} Similarity Matches"
    cv2.imshow(window_name, combined_image)
    
    # Wait for keypress or window closure
    while True:
        key = cv2.waitKey(100)  # Check every 100ms
        if key != -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
  
