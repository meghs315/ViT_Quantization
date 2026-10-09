# ViT Quantization: FP32 vs. Weight-Only INT8

## Overview

This repository recreates and adapts quantization experiments I worked on during research at SSRL, where quantization techniques were explored for a medical-imaging binary classification task.

For this project, I use a pretrained Vision Transformer (ViT-Base) and the CIFAR-10 dataset to investigate the trade-off between model checkpoint size and classification accuracy when applying symmetric, per-tensor, weight-only INT8 quantization.

The experiment establishes a baseline using FP32 weights, applies INT8 quantization to selected model weights, and evaluates the resulting model to measure the effect on accuracy and checkpoint size.

## Experiment Setup

* **Model:** Google ViT-Base (`google/vit-base-patch16-224`)
* **Dataset:** CIFAR-10, filtered to two classes: airplane and automobile
* **Task:** Binary image classification
* **Hardware:** Apple Silicon Mac, using PyTorch MPS
* **Quantization:** Symmetric, per-tensor, weight-only INT8
* **Evaluation set:** 2,000 images from the CIFAR-10 test set

The pretrained ViT backbone is frozen while the classification head is trained for the binary classification task. The FP32 model provides the baseline against which the quantized checkpoint is evaluated.

## Results

| Configuration              | Test Accuracy | Checkpoint Size |
| -------------------------- | ------------: | --------------: |
| FP32 baseline              |        98.85% |      327.37 MiB |
| Symmetric weight-only INT8 |        98.75% |       84.42 MiB |

The INT8 checkpoint is approximately **74.21% smaller** than the FP32 checkpoint. Test accuracy decreased by just **0.10 percentage points**.

These results suggest that the selected weights can be quantized with a small change in classification accuracy while substantially reducing serialized checkpoint size.

## Quantization Method

The quantization implementation uses symmetric, per-tensor quantization. Each selected weight tensor uses a scale calculated from its maximum absolute value:

`scale = max(abs(weights)) / 127`

Weights are rounded and clamped to the signed INT8 range of -127 to 127. The quantized integer weights and their corresponding scales are stored in the quantized checkpoint.

The implementation targets selected floating-point weight matrices, while leaving other parameters unchanged.

## Validation

The quantization validation script checks representative numerical examples, verifies quantized checkpoint structure and tensor shapes, and measures reconstruction error.

The validation run confirmed:

* 72 tensors quantized and 128 tensors left unchanged.
* 84,934,656 quantized weights checked.
* All validation checks passed.
* Mean absolute reconstruction error: 0.002739.
* Root mean squared reconstruction error: 0.003760.
* Maximum absolute reconstruction error: 0.014393.

The FP32 and quantized checkpoints were evaluated on the same 2,000 test images. The first five batches were excluded from latency measurements as warm-up, while all 2,000 images were included in accuracy calculations.

## Limitations and Future Work

**This experiment does not demonstrate native INT8 inference acceleration.** During evaluation, quantized weights are reconstructed into FP32 before inference. Consequently, the measured inference latency reflects FP32 execution rather than optimized INT8 computation.

The current results demonstrate checkpoint-size reduction and the accuracy impact of weight quantization, not reduced inference latency or runtime memory consumption.

Potential next steps include:

* Benchmarking on a cloud GPU.
* Comparing against a framework-supported INT8 inference implementation.
* Measuring inference latency, throughput, and runtime memory under equivalent conditions.
* Exploring additional quantization formats after validating the current implementation.

## How to Run

Run the following commands from the project root directory, with the required Python dependencies installed.

### 1. Train the FP32 baseline

```bash
python fine_tune_fp32.py
```

### 2. Evaluate the FP32 model

```bash
python evaluate.py
```

### 3. Quantize the model weights

```bash
python int8_quantize.py
```

### 4. Evaluate the quantized checkpoint

```bash
python evaluate_int8.py
```

### 5. Validate the quantization implementation

```bash
python validate_quantization.py
```

The validation script checks quantization behavior, checkpoint consistency, and numerical reconstruction error.


## References

1. Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” *International Conference on Learning Representations (ICLR)*, 2021. https://arxiv.org/abs/2010.11929

2. Jia Deng et al. “ImageNet: A Large-Scale Hierarchical Image Database.” *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009.

3. Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto, 2009. https://www.cs.toronto.edu/~kriz/cifar.html
