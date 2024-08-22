import re
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def find_image_path(card_id, root_dir="cards"):
    root_path = Path(root_dir)
    for image_path in root_path.rglob(f"{card_id}"):
        return image_path
    return None

def extract_prefix(card_id):
    match = re.match(r'^[^-]+', card_id)
    return match.group(0) if match else ''

class SingleSampleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.images_in_memory = []

        # Collect image paths and corresponding labels
        for series in os.listdir(root_dir):
            series_path = os.path.join(root_dir, series)
            if os.path.isdir(series_path):
                for img_name in os.listdir(series_path):
                    if ".json" in img_name:
                        continue
                    img_path = os.path.join(series_path, img_name)
                    self.image_paths.append(img_path)
                    
                    label = img_name  
                    if label not in self.label_to_idx:
                        idx = len(self.label_to_idx)
                        self.label_to_idx[label] = idx
                        self.idx_to_label[idx] = label
                    
                    self.labels.append(self.label_to_idx[label])

                    # Load the image into memory
                    image = Image.open(img_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    self.images_in_memory.append(image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.images_in_memory[idx]
        label_idx = self.labels[idx]
        one_hot_label = F.one_hot(torch.tensor(label_idx), num_classes=len(self.label_to_idx))

        if self.transform:
            image = self.transform(image)

        return image, one_hot_label.float()
    
    def set_preprocessing(self, preprocessing):
        self.transform = preprocessing


class CardModel(nn.Module):
    def __init__(self, num_labels):
        super(CardModel, self).__init__()
        #####
        ##### BACKBONE DEFINITION
        #####
        self.weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.transform = self.weights.transforms()
        self.base_model = models.mobilenet_v3_large(self.weights)
        num_ftrs = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Identity()
        embedding_size = 256

        self.embedding_layer = nn.Linear(num_ftrs, embedding_size)
        self.classification_layer = nn.Linear(embedding_size, num_labels)

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x) 
        logits = self.classification_layer(x)
        return x, logits

    def preprocess(self, image):
        return self.transform(image)
    
    def transform_info(self):
        mean = self.weights.transforms.func(crop_size=1).mean
        std = self.weights.transforms.func(crop_size=1).std
        crop_size = 224
        return mean, std, crop_size


def extract_embedding(model: CardModel, img_path, device):
    model.eval()
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = model.preprocess(image)
    with torch.no_grad():
        embedding, logits = model(image.unsqueeze(0).to(device))
    return F.normalize(embedding, p=2, dim=1).cpu().numpy().flatten(), logits
