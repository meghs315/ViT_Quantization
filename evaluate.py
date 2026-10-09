
import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

DATA_DIR = "./data"
CHECKPOINT = "./vit_cifar10_binary_fp32.pth"
MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 2
WARMUP_BATCHES = 5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class CIFAR10Binary(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [
            i for i, label in enumerate(dataset.targets)
            if label in (0, 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, torch.tensor(
            0 if label == 0 else 1, dtype=torch.long
        )

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_data = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)
test_loader = DataLoader(
    CIFAR10Binary(test_data),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

state = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True
)
model.load_state_dict(state)
model.to(device).eval()

correct = 0
total = 0
timings = []
warmup_count = 0
timed_images = 0

def synchronize():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Run inference for every batch, including warm-up batches.
        synchronize()
        start = time.perf_counter()
        outputs = model(pixel_values=images)
        synchronize()
        elapsed = time.perf_counter() - start

        # Every test image contributes to accuracy.
        predictions = outputs.logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Exclude warm-up batches from performance measurements only.
        if warmup_count < WARMUP_BATCHES:
            warmup_count += 1
            continue

        timings.append(elapsed)
        timed_images += labels.size(0)

if total == 0:
    raise RuntimeError("No test images were evaluated.")
if timed_images == 0:
    raise RuntimeError("No batches remain for timing.")

accuracy = correct / total
total_time = sum(timings)
mean_batch_latency = np.mean(timings)
mean_image_latency = total_time / timed_images
throughput = timed_images / total_time
checkpoint_mb = os.path.getsize(CHECKPOINT) / (1024 ** 2)

print(f"Device: {device}")
print(f"Evaluated images: {total}")
print(f"Timed images: {timed_images}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Mean batch latency: {mean_batch_latency * 1000:.2f} ms")
print(f"Mean image latency: {mean_image_latency * 1000:.2f} ms")
print(f"Throughput: {throughput:.2f} images/sec")
print(f"Checkpoint size: {checkpoint_mb:.2f} MiB")