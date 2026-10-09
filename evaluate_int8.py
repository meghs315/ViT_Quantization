import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"
FP32_PATH = "./vit_cifar10_binary_fp32.pth"
INT8_PATH = "./vit_cifar10_binary_int8.pt"
DATA_DIR = "./data"
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


def synchronize():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_model():
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    state = torch.load(
        FP32_PATH, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    return model


def reconstruct_int8(model, checkpoint_path):
    packed = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    state = packed["quantized_state_dict"]
    scales = packed["scales"]
    zero_points = packed["zero_points"]
    offsets = packed["offsets"]

    reconstructed = {}

    for name, q in state.items():
        if name in scales:
            scale = scales[name]
            zero_point = zero_points[name]
            offset = offsets[name]

            if packed["metadata"]["method"] == "asymmetric":
                if offset.item() != 0:
                    weight = q.float() * scale + offset
                else:
                    weight = (q.float() - zero_point.float()) * scale
            else:
                weight = q.float() * scale

            reconstructed[name] = weight
        else:
            reconstructed[name] = q

    model.load_state_dict(reconstructed)
    return model


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    agreements = 0
    timings = []
    warmups = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if warmups < WARMUP_BATCHES:
                model(pixel_values=images)
                synchronize()
                warmups += 1
                continue

            synchronize()
            start = time.perf_counter()
            logits = model(pixel_values=images).logits
            synchronize()
            timings.append(time.perf_counter() - start)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    elapsed = sum(timings)

    return {
        "accuracy": correct / total,
        "mean_batch_latency_ms": np.mean(timings) * 1000,
        "mean_image_latency_ms": elapsed / total * 1000,
        "throughput_images_per_second": total / elapsed,
        "evaluated_images": total,
    }


def main():
    for path in (FP32_PATH, INT8_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

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
    loader = DataLoader(
        CIFAR10Binary(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Device: {device}")

    print("\nEvaluating FP32...")
    fp32_model = load_model().to(device)
    fp32_results = evaluate(fp32_model, loader)
    del fp32_model

    print("\nEvaluating reconstructed INT8 weights...")
    int8_model = reconstruct_int8(
        load_model(), INT8_PATH
    ).to(device)
    int8_results = evaluate(int8_model, loader)

    print("\n--- Results ---")
    for name, result in (
        ("FP32", fp32_results),
        ("INT8 reconstructed", int8_results),
    ):
        print(f"\n{name}")
        for metric, value in result.items():
            print(f"{metric}: {value}")

    print(
        "\nAccuracy difference (percentage points): "
        f"{(int8_results['accuracy'] - fp32_results['accuracy']) * 100:.2f}"
    )


if __name__ == "__main__":
    main()