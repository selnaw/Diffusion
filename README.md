```markdown
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
```

2. Install core dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mani-skill3 gymnasium sapien h5py diffusers sympy wandb numpy
```

3. (Optional) Import environment directly from config file
```bash
conda env create -f environment.yml
```

### Dataset Preparation
- Task: `PickCube-v1` motion planning expert demonstrations
- Format: HDF5 trajectory files with JSON metadata
- Scale: 100 expert trajectories
- Default path: `~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5`
- Preprocessing: Automatic flattening of nested observation dict + action space normalization

### Training
#### 1. Train unconditional diffusion model (baseline)
```bash
python train_diffusion.py
```
Output model: `diffusion_policy_final.pt`

#### 2. Train conditional diffusion policy (full version)
```bash
python train_diffusion_integrated.py --task PickCube-v1 --epochs 120 --batch-size 256
```
Output model: `cond_best_optimized.pt`

### Visualization & Evaluation
#### 1. Visualize unconditional model
```bash
python visualize_unconditional.py
```

#### 2. Visualize conditional diffusion policy
```bash
python visualize_final.py --model-path cond_best_optimized.pt --episodes 3
```

## Project Structure
```
Diffusion/
├── model/                  # Model backbone & diffusion scheduler implementation
├── src/                    # Core scripts: preprocessing, training, evaluation
├── eval_videos/            # Saved simulation demo videos
├── train_diffusion.py      # Unconditional diffusion training script
├── train_diffusion_integrated.py  # Conditional diffusion full training pipeline
├── visualize_unconditional.py     # Unconditional model visualization
├── visualize_final.py      # Conditional policy simulation visualization
├── arm_demo.py             # Basic ManiSkill environment demo
├── environment.yml         # Conda environment config
└── README.md
```

## Development Workflow
> Complete iteration of requirement analysis → data processing → model building → tuning optimization
1. **Requirement Analysis**: Compare BC/IRL defects, select Diffusion Policy as technical route, determine ManiSkill3 PickCube-v1 as verification task
2. **Data Processing**: Parse HDF5 expert data, solve nested observation dict loading crash, design sliding window obs/action chunking, complete normalization
3. **Model Building**: First implement unconditional diffusion baseline; then add observation condition branch, build conditional diffusion policy; implement fallback scheme for diffusers import failure
4. **Tuning Optimization**: Adjust batch size & learning rate for CPU training; add gradient clipping, weight decay and cosine LR scheduler; add EMA to stabilize training
5. **Verification & Closing**: Build simulation visualization pipeline, verify convergence, summarize bottlenecks and propose optimization roadmap

## Core Implementation
### 1. Unconditional Diffusion Baseline
- Backbone: 4-layer MLP with Dropout
- Scheduler: Linear beta schedule, 200 diffusion steps
- Loss: MSE loss for noise prediction
- Optimizer: AdamW with weight decay

### 2. Conditional Diffusion Policy
- Input: Historical observation sequence + noised action chunk + timestep embedding
- Backbone: Priority ConditionalUnet1D, fallback to MLP
- Scheduler: Cosine beta schedule (squaredcos_cap_v2), 100 diffusion steps
- Chunking: Observation horizon=2, Action horizon=8, Prediction horizon=16
- Training trick: Gradient clipping, EMA, cosine learning rate warmup

## Engineering Challenges & Solutions
| Problem | Root Cause | Solution |
|---------|-----------|----------|
| HDF5 observation tensor conversion crash | Official loader cannot handle nested dict observations | Custom flatten module converts nested dict to fixed-dim float vector |
| diffusers import failure / ConditionalUnet1D unavailable | Environment version mismatch | Full manual implementation of DDPM forward/reverse process, replace UNet with MLP backbone |
| RTX 5060 CUDA-PyTorch incompatibility | Driver & version mismatch, GPU training disabled | Adjust batch size & learning rate, adapt to CPU computing constraints |
| Missing observation key in demo files | Inconsistent data format across versions | Add automatic fallback to environment state data |

## Experiment Results
### Convergence Performance
- Initial training loss: ~0.69
- Final converged loss: ~0.35
- Conclusion: The full pipeline runs normally, and the model can capture basic statistical features of expert action distribution

### Trajectory Performance
- Unconditional model: Outputs random joint trajectories without valid manipulation logic (only as baseline reference)
- Conditional MLP policy: Generates basic executable action sequences; trajectory smoothness is limited by MLP feature extraction capability
- Verified advantage: Diffusion policy naturally outputs temporally coherent action chunks and avoids mode collapse

## Limitations & Future Work
### Current Limitations
1. CPU-only training limits iteration times and hyperparameter tuning space
2. MLP denoiser has weaker feature extraction than standard ConditionalUnet1D
3. Lack of EMA, dynamic LR scheduling and other training stabilizers in baseline version
4. Imperfect observation flattening logic leads to unstable training gradients

### Optimization Roadmap
1. Fix CUDA-PyTorch version matching to enable RTX 5060 GPU acceleration
2. Replace MLP denoiser with standard 1D conditional UNet
3. Add EMA, LR scheduler and gradient clipping to stabilize training
4. Build automatic evaluation loop to record task success rate and average reward
5. Integrate 3D point cloud features to improve generalization in unseen environments

## License
MIT License - Open for academic research and secondary development.
```
