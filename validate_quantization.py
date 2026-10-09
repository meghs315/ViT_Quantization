
import torch

from int8_quantize import quantize_tensor

FP32_PATH = "./vit_cifar10_binary_fp32.pth"
INT8_PATH = "./vit_cifar10_binary_int8.pt"


def validate_quantization_math():
    """Test known values, zeros, integer range, and reconstruction error."""
    test_cases = [
        torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
        torch.zeros(8),
        torch.tensor([[0.01, -0.02], [0.03, -0.04]]),
    ]

    for index, original in enumerate(test_cases, start=1):
        quantized, scale = quantize_tensor(original)
        reconstructed = quantized.float() * scale

        assert quantized.dtype == torch.int8
        assert quantized.shape == original.shape
        assert torch.isfinite(scale).all()
        assert scale.item() > 0
        assert quantized.min().item() >= -127
        assert quantized.max().item() <= 127
        assert torch.isfinite(reconstructed).all()

        max_error = (original - reconstructed).abs().max().item()
        tolerance = scale.item() / 2 + 1e-6

        assert max_error <= tolerance, (
            f"Test {index}: reconstruction error exceeds "
            "the expected rounding tolerance."
        )

        print(
            f"Math test {index}: PASS | "
            f"max absolute error = {max_error:.8f}"
        )


def validate_saved_checkpoint():
    """Compare packed INT8 tensors against original FP32 weights."""
    original = torch.load(
        FP32_PATH, map_location="cpu", weights_only=True
    )
    packed = torch.load(
        INT8_PATH, map_location="cpu", weights_only=True
    )

    quantized_state = packed["quantized_state_dict"]
    scales = packed["scales"]

    assert packed["metadata"]["method"] == "symmetric"
    assert set(original) == set(quantized_state)
    assert set(scales).issubset(set(original))

    quantized_count = 0
    unchanged_count = 0
    absolute_errors = []
    max_error = 0.0

    for name, original_tensor in original.items():
        stored_tensor = quantized_state[name]

        if name in scales:
            quantized_count += 1

            assert stored_tensor.dtype == torch.int8, (
                f"{name} is not stored as INT8"
            )
            assert stored_tensor.shape == original_tensor.shape

            scale = scales[name]
            assert torch.isfinite(scale).all()
            assert scale.item() > 0

            reconstructed = stored_tensor.float() * scale
            reference = original_tensor.float()
            errors = (reference - reconstructed).abs()

            assert torch.isfinite(reconstructed).all()
            assert errors.max().item() <= scale.item() / 2 + 1e-6, (
                f"{name}: reconstruction error exceeds "
                "the expected rounding tolerance."
            )

            absolute_errors.append(errors.reshape(-1))
            max_error = max(max_error, errors.max().item())

        else:
            unchanged_count += 1

            assert stored_tensor.dtype == original_tensor.dtype
            assert torch.equal(stored_tensor, original_tensor), (
                f"Unquantized tensor changed: {name}"
            )

    if quantized_count == 0:
        raise AssertionError("No quantized tensors were found.")

    all_errors = torch.cat(absolute_errors)
    mae = all_errors.mean().item()
    rmse = all_errors.square().mean().sqrt().item()

    print("\nCheckpoint validation: PASS")
    print(f"Quantized tensors: {quantized_count}")
    print(f"Unchanged tensors: {unchanged_count}")
    print(f"Quantized weights checked: {all_errors.numel():,}")
    print(f"Mean absolute reconstruction error: {mae:.8f}")
    print(f"Root mean squared reconstruction error: {rmse:.8f}")
    print(f"Maximum absolute reconstruction error: {max_error:.8f}")


if __name__ == "__main__":
    validate_quantization_math()
    validate_saved_checkpoint()
    print("\nAll quantization validation checks passed.")