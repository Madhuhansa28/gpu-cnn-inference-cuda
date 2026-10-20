# GPU-Accelerated CNN Inference with Custom CUDA Kernels

![Status](https://img.shields.io/badge/status-ongoing-orange)
![Language](https://img.shields.io/badge/language-C%2FC%2B%2B%20%7C%20CUDA-blue)
![Field](https://img.shields.io/badge/field-High%20Performance%20Computing-green)

> **Status: Ongoing.** This repository documents the design and progress of an individual High Performance Computing project. Results will be added as each stage is completed.

## Overview

Running a convolutional neural network (CNN) is computationally heavy, because almost all of the time is spent in convolution and matrix multiplication. Deep learning libraries make this fast on GPUs, but they hide how.

This project implements CNN inference **from scratch in C/C++ and CUDA**, then optimizes the GPU kernels step by step. Each version is benchmarked for speed, scaling and accuracy against a serial CPU baseline and against PyTorch (cuDNN) on the same GPU, to show how much each optimization contributes and how close custom kernels can get to a vendor library.

## Objectives

- Implement CNN inference (convolution, matrix multiplication, activation, pooling) without ML libraries.
- Write and optimize custom CUDA kernels for the bottleneck operations.
- Measure the effect of each optimization: shared-memory tiling, memory coalescing, block-size tuning and batching.
- Measure the speed and accuracy trade-off of reduced precision (FP16).
- Compare custom kernels against PyTorch/cuDNN and explain the results with a roofline model.

## Implementation Plan

Each version adds one optimization on top of the previous one, so the gain from each technique can be measured separately.

| # | Version | Runs on | Technique |
|---|---|---|---|
| 1 | Serial baseline | CPU | Plain C loops; reference point for all speedups |
| 2 | Naive CUDA | GPU | One thread per output value |
| 3 | Optimized CUDA | GPU | Shared-memory tiling, memory coalescing, tuned block sizes |
| 4 | Reduced precision + batching | GPU | FP16 arithmetic and batched inference |

**Planned extensions:** OpenMP multicore CPU version, INT8 quantization, kernel fusion of convolution and activation.

## Reference Implementation

PyTorch running the same trained model on the same GPU, which uses NVIDIA cuDNN internally.

## Dataset and Model

- **Dataset:** CIFAR-10 (60,000 images, 10 classes)
- **Model:** a small CNN trained in PyTorch, with weights exported for the C/CUDA engine

## Hardware

A single NVIDIA GPU (Google Colab / Kaggle or a local GPU) and a standard multicore CPU for the serial baseline. The exact GPU, CPU, CUDA version and compiler flags will be listed with the results.

## Evaluation Metrics

- **Speed:** latency per image, throughput (images/second), speedup over the serial baseline
- **Scaling:** throughput as batch size increases
- **Quality:** top-1 accuracy on the CIFAR-10 test set for every version
- **Analysis:** roofline model showing whether each kernel is compute-bound or memory-bound

## Targets

- At least 50x throughput over the serial CPU baseline
- Less than 1% accuracy loss with FP16
- Custom kernels within 3x of PyTorch/cuDNN speed

## Roadmap

- [ ] Literature review and CUDA fundamentals
- [ ] Train and export the CNN model
- [ ] Serial CPU baseline
- [ ] Naive CUDA version
- [ ] Optimized CUDA version
- [ ] FP16 and batched inference
- [ ] Benchmarking against PyTorch/cuDNN
- [ ] Roofline analysis
- [ ] Research paper
- [ ] *(Extension)* OpenMP, INT8, kernel fusion

## Planned Repository Structure

```
├── model/         # PyTorch training script and exported weights
├── src/
│   ├── serial/    # CPU baseline
│   ├── cuda_naive/
│   ├── cuda_opt/  # Optimized kernels
│   └── fp16/      # Reduced precision version
├── benchmarks/    # Benchmark scripts and raw results
├── results/       # Graphs and analysis
└── docs/          # Reports and paper
```

## Technologies

C/C++ · CUDA · OpenMP · PyTorch · Python · Matplotlib

## Author

**M.D.M. Wijesinghe**
B.Sc.Eng. Computer Engineering, University of Sri Jayewardenepura
