# Full-Stack Implementation of Robotic Arm Grasping Based on Diffusion Policy
> Course Project: CS461 (EX2) Robotics Final Project
>
> End-to-end Diffusion Policy imitation learning pipeline for robotic block grasping on the ManiSkill3 simulation platform

## Table of Contents
- [Project Overview](#project-overview)
- [Core Highlights](#core-highlights)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
  - [Environment Requirements](#environment-requirements)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Visualization & Evaluation](#visualization--evaluation)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Core Implementation](#core-implementation)
- [Engineering Challenges & Solutions](#engineering-challenges--solutions)
- [Experiment Results](#experiment-results)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

## Project Overview
Traditional imitation learning algorithms (Behavior Cloning, IRL) suffer from covariate shift, mode collapse and poor generalization. This project implements conditional & unconditional Diffusion Policy for the `PickCube-v1` grasping task on ManiSkill3, builds a complete offline imitation learning pipeline from data parsing to model training and simulation evaluation, and fixes multiple compatibility bugs in official baseline scripts. It verifies that diffusion-based visuomotor policies can generate temporally continuous action trajectories and effectively avoid mode collapse.

## Core Highlights
1. **Self-implemented full DDPM pipeline**: Complete forward/reverse denoising process implemented manually, no dependency on official `diffusers` DDPMScheduler
2. **Fixed official dataset crash**: Custom flattening module solves nested dictionary observation → tensor conversion error in ManiSkill3 baseline
3. **Dual model variants**: Both unconditional diffusion baseline and conditional diffusion policy with observation input are provided
4. **Full pipeline available**: Supports end-to-end data preprocessing, model training, simulation visualization and result export
5. **Fallback compatible design**: Automatically degrades to MLP backbone + custom scheduler when official UNet/diffusers are unavailable

## Tech Stack
| Category | Details |
|---------|--------|
| Language | **Python 3.10** |
| Core Framework | **PyTorch 2.5.1 (CUDA 11.8)** |
| Simulation Engine | ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium |
| Diffusion Components | diffusers 0.37.1 (fallback available) |
| Data Processing | h5py, NumPy, SymPy |
| Experiment Logging | Weights & Biases (wandb) |
| Environment Management | Miniconda (env name: `robodiff`) |

> Note: RTX 5060 Laptop has compatibility issues with current PyTorch CUDA version; all experiments can run normally on CPU.

## Quick Start
### Environment Requirements
1. Create and activate conda environment
```bash
conda create -n robodiff python=3.10
conda activate robodiff
