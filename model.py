import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from PIL import Image


class CardModel(nn.Module):
    def __init__(self, num_labels, feature_dim=256, pretrained=True):
        super(CardModel, self).__init__()
        #####################################
        #####    Backbone definition   ######
        #####################################
        self.weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.transform = self.weights.transforms() # use IMAGENET1K_V2 pre-processing directly
        self.base_model = models.mobilenet_v3_large(weights=self.weights if pretrained else None)
        self.save_name_string = "mobilenet_v3_large"
        num_ftrs = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Identity()
        self.feature_layer = nn.Linear(num_ftrs, feature_dim)
        self.classification_layer = nn.Linear(feature_dim, num_labels)

    def forward(self, x):
        x = self.base_model(x)
        features = self.feature_layer(x)
        logits = self.classification_layer(features)
        return features, logits

    def preprocess(self, image):
        return self.transform(image)
    
    # Used for training
    def transform_info(self):
        return self.transform.mean, self.transform.std, self.transform.resize_size, self.transform.crop_size
    
    def save_name(self):
        return self.save_name_string

    def classify_image(self, img, device):
        self.eval()
        if isinstance(img, str):
            img = cv2.imread(img)   
        elif isinstance(img, Path):
            img = cv2.imread(img.resolve())     
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        image = self.preprocess(image)
        with torch.no_grad():
            _, logits = self(image.unsqueeze(0).to(device))
        return np.squeeze(torch.softmax(logits, dim=1).detach().cpu().numpy())
