import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import timer


# --------------------------------------------------
# 1. Load CIFAR-10
# --------------------------------------------------

def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


batch = unpickle("data/cifar-10-batches-py/test_batch")

X = batch[b"data"]
y = np.array(batch[b"labels"])

# CIFAR-10 stores images as:
# [R1...R1024, G1...G1024, B1...B1024]
# Convert to:
# (N, 32, 32, 3)

X = X.reshape(-1, 3, 32, 32)
X = X.transpose(0, 2, 3, 1)

print("Images:", X.shape)
print("Labels:", y.shape)


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# --------------------------------------------------
# 3. Load ViT-Base
# --------------------------------------------------

model_name = "google/vit-base-patch16-224"

processor = ViTImageProcessor.from_pretrained(model_name)

model = ViTForImageClassification.from_pretrained(
    model_name
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)


# --------------------------------------------------
# 4. DataLoader
# --------------------------------------------------

dataset = CIFAR10Dataset(X, y, processor)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

# --------------------------------------------------
# 5. Evaluate accuracy + latency
# --------------------------------------------------

import time

correct = 0
total = 0

latencies = []

with torch.no_grad():

    for batch in loader:

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # CUDA operations are asynchronous, so synchronize
        # before starting the timer.
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        outputs = model(pixel_values=pixel_values)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        # Batch latency in milliseconds
        batch_latency_ms = (end - start) * 1000
        latencies.append(batch_latency_ms)

        predictions = outputs.logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)


# --------------------------------------------------
# Results
# --------------------------------------------------

accuracy = 100 * correct / total

# Average batch latency
avg_batch_latency_ms = np.mean(latencies)

# Latency per image
avg_latency_per_image_ms = (
    avg_batch_latency_ms / loader.batch_size
)

# Throughput
throughput = 1000 / avg_latency_per_image_ms

print(f"Accuracy:              {accuracy:.2f}%")
print(f"Avg batch latency:     {avg_batch_latency_ms:.2f} ms")
print(f"Avg image latency:     {avg_latency_per_image_ms:.2f} ms")
print(f"Throughput:            {throughput:.2f} images/sec")
