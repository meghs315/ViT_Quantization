
import argparse
import os
import torch
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"

SENSITIVE_LAYERS = (
    "norm",
    "embeddings",
    "classifier",
    "head",
    "cls_token",
)


def should_quantize(name, tensor):
    """Quantize matrix weights, excluding sensitive components."""
    if any(layer in name.lower() for layer in SENSITIVE_LAYERS):
        return False

    # Keep biases, vectors, and non-floating tensors unchanged.
    return (
        name.endswith(".weight")
        and tensor.ndim >= 2
        and tensor.is_floating_point()
    )

def quantize_tensor(tensor):
    """Symmetric per-tensor quantization to signed INT8."""
    x = tensor.detach().to(dtype=torch.float32)

    if not torch.isfinite(x).all():
        raise ValueError("Cannot quantize a tensor with NaN or infinity.")

    qmin, qmax = -127, 127
    max_abs = x.abs().max().item()

    if max_abs == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = torch.tensor(max_abs / qmax, dtype=torch.float32)

    quantized = torch.round(x / scale).clamp(qmin, qmax)
    return quantized.to(torch.int8), scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./vit_cifar10_binary_fp32.pth",
    )
    parser.add_argument(
        "--output",
        default="./vit_cifar10_binary_int8.pt",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    print("Loading FP32 model...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    state = torch.load(
        args.input,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    state = model.state_dict()

    quantized_state = {}
    scales = {}
    quantized_count = 0
    skipped_count = 0

    print(f"Quantization method: symmetric")

    for name, tensor in state.items():
        if should_quantize(name, tensor):
            q, scale = quantize_tensor(tensor)

    output = {
        "quantized_state_dict": quantized_state,
        "scales": scales,
        "metadata": {
            "base_model": MODEL_NAME,
            "method": "symmetric",
            "quantized_tensors": quantized_count,
            "unchanged_tensors": skipped_count,
            "quantization": "per-tensor weight-only INT8",
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(output, args.output)

    size_mb = os.path.getsize(args.output) / (1024 ** 2)

    print(f"Quantized tensors: {quantized_count}")
    print(f"Unchanged tensors: {skipped_count}")
    print(f"Saved checkpoint: {args.output}")
    print(f"Checkpoint size: {size_mb:.2f} MiB")
    print(
        "Note: This is a packed quantization checkpoint, "
        "not a directly executable Hugging Face model."
    )


if __name__ == "__main__":
    main()