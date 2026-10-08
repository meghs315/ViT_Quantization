import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import time
import random
import numpy as np

DATA_DIR = "./data"
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_PATH = "./vit_cifar10_binary_fp32.pth"

BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class CIFAR10Binary(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        # CIFAR-10: airplane = 0, automobile = 1
        self.indices = [
            i
            for i, label in enumerate(base_dataset.targets)
            if label in (0, 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_dataset[self.indices[idx]]

        # Keep binary labels: airplane = 0, automobile = 1
        binary_label = 0 if label == 0 else 1

        return image, torch.tensor(
            binary_label,
            dtype=torch.long,
        )

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

print("Loading CIFAR-10...")

cifar10_train = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

cifar10_test = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

train_dataset = CIFAR10Binary(cifar10_train)
test_dataset = CIFAR10Binary(cifar10_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

print("Loading pretrained ViT...")

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

model = model.to(device)

print("Model loaded.")
print(
    f"Number of parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

# Freeze the pretrained ViT backbone so only the new classifier is trained.
for parameter in model.vit.parameters():
    parameter.requires_grad = False

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in trainable_parameters):,}"
)

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)

for epoch in range(NUM_EPOCHS):
    model.train()

    # The ViT backbone remains frozen, but the classifier must be trainable.
    model.vit.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=images,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (batch_idx + 1) % 250 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    elapsed = time.perf_counter() - start_time

    train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / total

    print()
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"completed in {elapsed:.2f}s"
    )
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print()

model.eval()

correct = 0
total = 0

start_time = time.perf_counter()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(
            pixel_values=images
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

elapsed = time.perf_counter() - start_time

test_accuracy = correct / total

print()
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test inference time: {elapsed:.2f}s")

torch.save(
    model.state_dict(),
    OUTPUT_PATH,
)

print()
print(f"FP32 checkpoint saved to: {OUTPUT_PATH}")