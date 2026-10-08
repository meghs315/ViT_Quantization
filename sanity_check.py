import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification


DATA_DIR = "./data"
MODEL_NAME = "google/vit-base-patch16-224"

BATCH_SIZE = 8
NUM_CLASSES = 2


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


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

cifar10_test = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

binary_test = CIFAR10Binary(cifar10_test)

test_loader = DataLoader(
    binary_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Total CIFAR-10 test images: {len(cifar10_test)}")
print(f"Binary test images: {len(binary_test)}")


airplane_count = 0
automobile_count = 0

for _, label in binary_test:
    if label.item() == 0:
        airplane_count += 1
    else:
        automobile_count += 1

print(f"Airplane: {airplane_count}")
print(f"Automobile: {automobile_count}")


print("Loading pretrained ViT...")

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)

model = model.to(device)
model.eval()

print("Model loaded.")
print(f"Number of classes: {model.config.num_labels}")
print(f"Classifier outputs: {model.classifier.out_features}")


images, labels = next(iter(test_loader))

print(f"Input shape: {images.shape}")
print(f"Labels: {labels.tolist()}")

images = images.to(device)
labels = labels.to(device)


print("Running forward pass...")

with torch.no_grad():
    outputs = model(pixel_values=images)

logits = outputs.logits

predictions = torch.argmax(
    logits,
    dim=1,
)

probabilities = torch.softmax(
    logits,
    dim=1,
)


print(f"Logits shape: {logits.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")

print("Logits:")
print(logits)

print("Predictions:")
print(predictions)

print("Probabilities:")
print(probabilities)


assert images.shape == (
    BATCH_SIZE,
    3,
    224,
    224,
)

assert labels.shape == (
    BATCH_SIZE,
)

assert logits.shape == (
    BATCH_SIZE,
    NUM_CLASSES,
)

assert predictions.shape == (
    BATCH_SIZE,
)

assert torch.all(
    (labels == 0) | (labels == 1)
)


print()
print("Sanity check passed.")
print("No training was performed.")