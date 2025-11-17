import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "model.pth")

# Labels in alphabetical order (ImageFolder)
CLASSES = ["paper", "rock", "scissors"]


def _build_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


MODEL = _build_model()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    img = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        output = MODEL(img)
        _, predicted = torch.max(output, 1)
        label = CLASSES[predicted.item()]
    return label
