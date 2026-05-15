import cv2
import numpy as np
import torch
from typing import Tuple
from util import PROJECT_ROOT, DATASET_ROOT_DIR, set_seed, extract_prefix
from model import CardModel


SEED = 42
set_seed(SEED)
STATE_DICT = PROJECT_ROOT / 'mobile_large_v1_state_dict.pth'

classes = None
with open(DATASET_ROOT_DIR / 'names.txt', 'r') as file:
    classes = file.read().splitlines()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using DEVICE=", DEVICE)

model = CardModel(num_labels=len(classes), pretrained=False).eval()
model.to(DEVICE)
state_dict = torch.load(STATE_DICT, weights_only=True, map_location=DEVICE)
state_dict = {key.replace("embedding_layer.", "feature_layer."): value for key, value in state_dict.items()}
model.load_state_dict(state_dict)

def classify_image(img, min_conf_threshold=0.5) -> Tuple[str | None, float]:
    probs = model.classify_image(img, DEVICE)
    predicted_class_idx = np.argmax(probs)
    predicted_class_conf = probs[predicted_class_idx]
    
    if predicted_class_conf > min_conf_threshold:
        return classes[predicted_class_idx], predicted_class_conf
    else:
        return None, predicted_class_conf

if __name__ == "__main__":
    predicted_class_name, confidence = classify_image(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png')
    print(predicted_class_name)
    if predicted_class_name is None:
        raise Exception("No match for classification method")
    
    img = cv2.imread(DATASET_ROOT_DIR / 'cards' / extract_prefix(predicted_class_name) / predicted_class_name)
    cv2.imshow(f"Classification method; Predicted card: {predicted_class_name}", img)
    cv2.waitKey(1000)
