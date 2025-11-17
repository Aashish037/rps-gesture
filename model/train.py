import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# -----------------------------
# PATHS / CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # /model
DATASET_DIR = os.path.join(BASE_DIR, "dataset")           # /model/dataset
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")          # /model/saved_model
SAVE_PATH = os.path.join(SAVE_DIR, "model.pth")


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    num_workers: int = 0  # windows safe default


CONFIG = TrainConfig()

print("BASE_DIR:", BASE_DIR)
print("DATASET_DIR:", DATASET_DIR)

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------
COMMON_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    COMMON_NORMALIZE,
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    COMMON_NORMALIZE,
])

# -----------------------------
# LOAD DATASET & SPLIT
# -----------------------------
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
num_val = int(len(full_dataset) * CONFIG.val_split)
num_train = len(full_dataset) - num_val
train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
val_dataset.dataset.transform = val_transform  # ensure eval transform

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=True,
    num_workers=CONFIG.num_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=False,
    num_workers=CONFIG.num_workers,
)

print("Classes found:", full_dataset.classes)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# -----------------------------
# MODEL
# -----------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# TRAINING SETUP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def run_epoch(loader, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total else 0
    accuracy = correct / total if total else 0
    return avg_loss, accuracy


best_val_acc = 0
os.makedirs(SAVE_DIR, exist_ok=True)

print("\nStarting Training...\n")
for epoch in range(1, CONFIG.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    scheduler.step()

    print(
        f"Epoch {epoch}/{CONFIG.epochs} "
        f"- train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} "
        f"- val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ↳ New best model saved (val_acc={val_acc:.3f})")

print("\nTraining complete!")
print(f"Best val accuracy: {best_val_acc:.3f}")
print(f"Model saved to: {SAVE_PATH}")
