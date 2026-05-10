import os
import cv2
import numpy as np
import sys
import torch 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from time import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import CardModel
from util import set_seed

SEED = 42
set_seed(SEED)

BATCH_SIZE = 32
NUM_WORKERS = 0
TRAIN_EPOCHS = 30
TRAIN_FINE_EPOCHS = 15
ITERATION = "v3"
DATASET_ROOT_DIR = PROJECT_ROOT / 'datasets' / 'card_recognizer'
EXPERIMENT_FOLDER = int(time())

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

class SingleSampleDataset(Dataset):
    def __init__(self, dataset_root_dir):
        self.dataset_root_dir = dataset_root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.images_in_memory = []

        # Collect image paths and corresponding labels
        for series in os.listdir(dataset_root_dir):
            series_path = os.path.join(dataset_root_dir, series)
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


dataset = SingleSampleDataset(dataset_root_dir=DATASET_ROOT_DIR / 'cards')
classes_list = list(dataset.label_to_idx.keys())
nr_classes = len(classes_list)
with open(DATASET_ROOT_DIR / 'names.txt', 'w') as file:
    for key in classes_list[:-1]:
        file.write(key + '\n')
    file.write(key)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
model = CardModel(nr_classes).to(DEVICE)

mean, std, resize_size, crop_size = model.transform_info()
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.RandomResizedCrop(crop_size, scale=(1, 1)),
    transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.96, 1.06)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    transforms.Normalize(mean=mean, std=std)
])
dataset.set_preprocessing(transform)

def train_loop(num_epochs, optimizer):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, one_hot_labels in train_loader:
            inputs = inputs.to(DEVICE)
            one_hot_labels = one_hot_labels.to(DEVICE)

            optimizer.zero_grad()
            
            _, logits = model(inputs)
            
            labels = torch.argmax(one_hot_labels, dim=1)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
        
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')


# Freeze all layers
for param in model.base_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loop(TRAIN_EPOCHS, optimizer)

# Unfreeze more layers and fine-tune with a lower learning rate
for param in model.base_model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_loop(TRAIN_FINE_EPOCHS, optimizer)

experiment_folder_path = Path(f'runs/{EXPERIMENT_FOLDER}/')
# Create the folder (and parent directories if needed)
experiment_folder_path.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), experiment_folder_path / f'{model.save_name()}_{ITERATION}_state_dict.pth')
torch.save(model, experiment_folder_path / f'{model.save_name()}_{ITERATION}.pth')

_, probs = model.extract_embedding(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png', DEVICE)
predicted_class = int(np.argmax(probs))
predicted_class_name = classes_list[predicted_class]
print(predicted_class_name)
