# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn.functional as F

from model import  CardModel, set_seed, extract_prefix, extract_embedding, find_image_path


SEED = 42
set_seed(SEED)
ROOT_DIR = "cards"

classes = None
with open('names.txt', 'r') as file:
    classes = file.read().splitlines()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

model = CardModel(num_labels=len(classes)).to(DEVICE).eval()
model.load_state_dict(torch.load("1724327867.455931_model_state_dict.pth", weights_only=True))

def infere(path):
    _, logits = extract_embedding(model, path, DEVICE)
    predicted_class = torch.argmax(F.softmax(logits, dim=1), dim=1).detach() #.item() if .detach() not working
    return classes[predicted_class]


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

    for series in tqdm(os.listdir(ROOT_DIR)):
        series_path = os.path.join(ROOT_DIR, series)
        if os.path.isdir(series_path):
            for img_name in os.listdir(series_path):
                if ".json" in img_name:
                    continue
                img_path = os.path.join(series_path, img_name)
                label = img_name  
                embedding, _ = extract_embedding(model, img_path, DEVICE)
                card_embeddings.append(embedding)
                index_to_card_id[index] = label
                index += 1

    card_embeddings_np = np.array(card_embeddings, dtype='float32')

def similarity_search(path, top_k=5):
    # Define a query vector
    query_embedding, _ = extract_embedding(model, path, DEVICE)
    query_embedding = np.array([query_embedding]) 

    distances = euclidean_distances(card_embeddings_np, query_embedding)

    closest_indices = np.argsort(distances.flatten())[:top_k]
    labels = [index_to_card_id[idx] for idx in closest_indices]
    distances = [distances[idx][0] for idx in closest_indices]
    return labels, distances


if __name__ == "__main__":
    predicted_class_name = infere("monkey.png")
    print(predicted_class_name)
    img = Image.open(ROOT_DIR + "/" +  extract_prefix(predicted_class_name) + "/" + predicted_class_name)
    img.show()

    print("Creating template db...")
    create_template_db()
    print("Creating done!")
    top_k = 5
    labels, distances = similarity_search("monkey.png", top_k)
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    if top_k == 1:
        axes = [axes] 

    for i, card_id in enumerate(labels):
        image_path = find_image_path(card_id, ROOT_DIR)
        if image_path:
            image = Image.open(image_path)
            print(f"Match {i + 1}:")
            print(f"Card ID: {card_id}")
            print(f"Distance: {distances[i]}")
            plt.imshow(image)
            plt.axis('off')
            plt.show()
# %%
