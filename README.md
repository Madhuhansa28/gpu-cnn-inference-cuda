# Parallel and Quantized CNN Inference on Edge and GPU Hardware

![Status](https://img.shields.io/badge/status-ongoing-orange)
![Language](https://img.shields.io/badge/language-C%2FC%2B%2B%20%7C%20CUDA-blue)
![Field](https://img.shields.io/badge/field-High%20Performance%20Computing-green)

> **Status: Ongoing.** This repository documents the design and progress of an individual High Performance Computing project. Results will be added as each stage is completed.

## Overview

Running a convolutional neural network (CNN) on low-power hardware is slow, because almost all of the time is spent in convolution and matrix multiplication. Libraries such as TensorFlow Lite make this faster but hide how they do it.

This project builds a small CNN inference engine **from scratch in C/C++** and speeds it up step by step using different levels of parallelism. Each version is benchmarked for speed, scaling, accuracy and energy use on three platforms, from a Raspberry Pi to an NVIDIA GPU.

## Objectives

- Implement CNN inference (convolution, matrix multiplication, activation, pooling) without ML libraries.
- Parallelize the bottleneck operations at every hardware level: cache, multicore, SIMD and GPU.
- Measure the speedup and accuracy trade-off of INT8 quantization.
- Compare each version against a reference implementation (TensorFlow Lite / ONNX Runtime).
- Analyze scaling, parallel efficiency and energy efficiency across hardware.

## Implementation Plan

Each version adds one optimization on top of the previous one, so the gain from each technique can be measured separately.

| # | Version | Technique |
|---|---|---|
| 1 | Serial baseline | Plain C loops; reference point for all speedups |
| 2 | Cache-optimized | Loop tiling and blocking |
| 3 | Multicore | OpenMP threads |
| 4 | SIMD | ARM NEON (Raspberry Pi) and AVX (x86 CPU) |
| 5 | Quantized | INT8 integer arithmetic |
| 6 | GPU | CUDA kernels with shared memory and batched inference |

## Hardware

| Platform | Purpose |
|---|---|
| Raspberry Pi 4/5 (4 ARM cores, NEON) | Edge device |
| Multicore x86 CPU (AVX) | Desktop/laptop comparison |
| NVIDIA GPU | GPU acceleration |

Full specifications and compiler flags will be listed with the results.

## Dataset and Model

- **Dataset:** CIFAR-10 (60,000 images, 10 classes)
- **Model:** a small CNN trained in PyTorch, with weights exported for the C/C++ engine

## Evaluation Metrics

- **Speed:** latency per image, throughput (images/second), speedup over the serial baseline
- **Scaling:** speedup and parallel efficiency for 1, 2, 4 and 8 threads; GPU throughput by batch size
- **Quality:** top-1 accuracy on the CIFAR-10 test set for every version
- **Energy:** images per joule on the Raspberry Pi
- **Analysis:** roofline model to show whether each version is compute-bound or memory-bound

## Targets

- At least 3x speedup over the serial baseline on a 4-core Raspberry Pi
- At least 20x throughput on the GPU compared with a single CPU core
- Less than 1% accuracy loss from INT8 quantization

## Roadmap

- [ ] Literature review
- [ ] Train and export the CNN model
- [ ] Serial baseline
- [ ] Cache-optimized version
- [ ] OpenMP multicore version
- [ ] SIMD version (NEON / AVX)
- [ ] INT8 quantized version
- [ ] CUDA GPU version
- [ ] Benchmarking and scaling experiments
- [ ] Energy measurements and roofline analysis
- [ ] Research paper

## Planned Repository Structure

```
├── model/         # PyTorch training script and exported weights
├── src/
│   ├── serial/    # Baseline implementation
│   ├── openmp/    # Multicore version
│   ├── simd/      # NEON / AVX version
│   ├── int8/      # Quantized version
│   └── cuda/      # GPU version
├── benchmarks/    # Benchmark scripts and raw results
├── results/       # Graphs and analysis
└── docs/          # Reports and paper
```

## Technologies

C/C++ · OpenMP · CUDA · ARM NEON · AVX · PyTorch · TensorFlow Lite · Python · Matplotlib · Raspberry Pi

## Author

**M.D.M. Wijesinghe**
B.Sc.Eng. Computer Engineering, University of Sri Jayewardenepura
