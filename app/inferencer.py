import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics.pairwise import euclidean_distances
from app.util import PROJECT_ROOT, DATASET_ROOT_DIR, set_seed, extract_prefix, display_top_k_images
from app.model import  CardModel


SEED = 42
set_seed(SEED)
STATE_DICT = PROJECT_ROOT / 'app' / 'weights' / 'mobilenet_v3_large_v3_state_dict.pth'

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

model = CardModel(num_labels=len(classes)).eval()
model.to(DEVICE)
model.load_state_dict(torch.load(STATE_DICT, weights_only=True, map_location=DEVICE))

def classify_image(img, min_conf_threshold=0.5) -> Tuple[str | None, float]:
    _, probs = model.extract_embedding(img, DEVICE)
    predicted_class_idx = np.argmax(probs)
    predicted_class_conf = probs[predicted_class_idx]
    
    if predicted_class_conf > min_conf_threshold:
        return classes[predicted_class_idx], predicted_class_conf
    else:
        return None, predicted_class_conf

#####################################
##### Template based approach  ######
#####################################

card_embeddings = []
index = 0
index_to_card_id = {}
card_embeddings_np = None

def create_template_db():
    global index_to_card_id
    global card_embeddings
    global index
    global card_embeddings_np

    for series in tqdm(os.listdir(DATASET_ROOT_DIR / 'cards')):
        series_path = os.path.join(DATASET_ROOT_DIR / 'cards', series)
        if os.path.isdir(series_path):
            for img_name in os.listdir(series_path):
                if ".json" in img_name:
                    continue
                img_path = os.path.join(series_path, img_name)
                label = img_name  
                embedding, _ = model.extract_embedding(img_path, DEVICE)
                card_embeddings.append(embedding)
                index_to_card_id[index] = label
                index += 1

    card_embeddings_np = np.array(card_embeddings, dtype='float32')

def similarity_search(path, top_k=5):
    # Define a query vector
    query_embedding, _ = model.extract_embedding(path, DEVICE)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances = euclidean_distances(card_embeddings_np, query_embedding)
    closest_indices = np.argsort(distances.flatten())[:top_k]
    labels = [index_to_card_id[idx] for idx in closest_indices]
    distances = [distances[idx][0] for idx in closest_indices]
    return labels, distances

  

if __name__ == "__main__":
    predicted_class_name, confidence = classify_image(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png')
    print(predicted_class_name)
    if predicted_class_name is None:
        raise Exception("No match for classification method")
    
    img = cv2.imread(DATASET_ROOT_DIR / 'cards' / extract_prefix(predicted_class_name) / predicted_class_name)
    cv2.imshow(f"Classification method; Predicted card: {predicted_class_name}", img)
    cv2.waitKey(1000)
    
    print("Creating template db for similarity search method...")
    create_template_db()
    print("Creating done!")
    top_k = 5
    labels, distances = similarity_search(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png', top_k)
    
    # Print and display top-k results
    print("\nSimilarity Search Results:")
    for i, card_id in enumerate(labels):
        print(f"Match {i + 1}:")
        print(f"Card ID: {card_id}")
        print(f"Distance: {distances[i]:.4f}")
    
    display_top_k_images(labels, distances, top_k)