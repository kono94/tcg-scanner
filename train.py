# %%
import time
import torch 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from model import SingleSampleDataset, CardModel, set_seed, extract_embedding

SEED = 42
set_seed(SEED)

BATCH_SIZE = 32
NUM_WORKERS = 0
ROOT_DIR = "cards"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

dataset = SingleSampleDataset(root_dir=ROOT_DIR)
classes_list = list(dataset.label_to_idx.keys())
nr_classes = len(classes_list)
with open('names.txt', 'w') as file:
    for key in classes_list[:-1]:
        file.write(key + '\n')
    file.write(key)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
model = CardModel(nr_classes).to(DEVICE)

mean, std, crop_size = model.transform_info()
transform = transforms.Compose([
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


# %%
# Freeze all layers
for param in model.base_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loop(30, optimizer)

# %%
# Unfreeze more layers and fine-tune with a lower learning rate
for param in model.base_model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_loop(15, optimizer)

torch.save(model.state_dict(), f'{time.time()}_model_state_dict.pth')
torch.save(model, f'{time.time()}_model_save.pth')

# %%
embedding, logits = extract_embedding(model, "monkey.png", DEVICE)
predicted_class = torch.argmax(F.softmax(logits, dim=1), dim=1).detach() #.item() if .detach() not working
predicted_class_name = classes_list[predicted_class]
print(predicted_class_name)
