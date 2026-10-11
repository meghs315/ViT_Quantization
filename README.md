# ViT Quantization: FP32 vs. Weight-Only INT8

## Overview

This repository recreates and adapts quantization experiments I worked on during research at SSRL, where quantization techniques were explored for a medical-imaging binary classification task. The presentation (pdf file) is from when I presented those findings. 

For this project, I use a pretrained Vision Transformer (ViT-Base) and the CIFAR-10 dataset to investigate the trade-off between model checkpoint size and classification accuracy when applying symmetric, per-tensor, weight-only INT8 quantization.

I also benchmarked FP32 vs. FP16 on a Colab T4: FP16 cut peak memory by about 45% and latency by 2.5-4.7x depending on batch size, which shows that checkpoint-size savings and runtime savings come from different techniques.

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

The current results demonstrate checkpoint-size reduction and the accuracy impact of weight quantization, not reduced inference latency or runtime memory consumption. A separate FP32 vs. FP16 GPU benchmark is below.

Potential next steps include:

* Comparing against a framework-supported INT8 inference implementation with the benchmark framework. 
* Measuring inference latency, throughput, and runtime memory under equivalent conditions.
* Exploring additional quantization formats after validating the current implementation.

## Runtime Benchmark (FP32 vs. FP16)

The INT8 work above reduces checkpoint size, but the weights are expanded back to FP32 at runtime, so it does not change GPU inference. To see what does, I benchmarked FP32 against FP16 on the same model.

| Batch | Precision | Latency (median) | Throughput | Peak GPU memory |
| ----: | --------- | ---------------: | ---------: | --------------: |
| 1  | FP32 | 13.7 ms  | 73 img/s  | 347 MiB |
| 1  | FP16 | 5.4 ms   | 184 img/s | 192 MiB |
| 8  | FP32 | 75.9 ms  | 105 img/s | 399 MiB |
| 8  | FP16 | 16.5 ms  | 485 img/s | 218 MiB |
| 32 | FP32 | 304.1 ms | 105 img/s | 580 MiB |
| 32 | FP16 | 64.1 ms  | 499 img/s | 308 MiB |

FP16 reduced peak memory by about 45% and latency by 2.5x (batch 1) to 4.7x (batch 32). The gain beyond 2x is likely due to the T4's FP16 hardware, though I did not test this.

| Technique | What it changes | Evidence |
| --------- | --------------- | -------- |
| Weight-only INT8 (stored, expanded to FP32 at runtime) | Checkpoint size on disk | 74% smaller, 0.10 pt accuracy change |
| FP16 | Runtime memory, latency, throughput | See table above |
| Native INT8 inference (TensorRT, bitsandbytes, etc.) | Potentially speed and runtime memory | Not tested; listed as future work |

**Setup:** Colab Tesla T4; pretrained `google/vit-base-patch16-224` (not the fine-tuned model); random 224x224 inputs; batch sizes 1, 8, 32; median of 50 timed runs after 10 warmup passes, with `torch.cuda.synchronize()` around each run; one model on the GPU at a time. Memory is peak PyTorch allocation (`max_memory_allocated`), not total GPU usage. Accuracy was not re-evaluated in FP16.

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
