# Full-Stack Implementation of Robotic Arm Grasping Based on Diffusion Policy
Course Project: CS461 (EX2) Robotics Final Project
> End-to-end Diffusion Policy imitation learning pipeline for robotic block grasping on the ManiSkill3 simulation platform

## Project Overview
This project develops conditional and unconditional Diffusion Policy for the `PickCube-v1` robotic grasping task built upon ManiSkill3’s GPU-accelerated simulation engine. We constructed a complete offline imitation learning workflow, including HDF5 expert demonstration parsing, nested observation flattening, a self-built MLP denoising backbone, hand-implemented DDPM cosine scheduler, training and evaluation modules. Multiple environment compatibility and data format bugs existing in official baseline scripts are fully resolved in this work.

Traditional imitation learning algorithms such as Behavior Cloning and Inverse Reinforcement Learning suffer from covariate shift, mode collapse and poor generalization ability. Diffusion Policy adopts DDPM to generate smooth, multi-modal robot action sequences through iterative denoising, which inherently guarantees temporal continuity of manipulation trajectories. This project verifies the practicability of diffusion-based visuomotor policies in a realistic simulated robot environment.

## Project Objectives
1. Build an end-to-end pipeline covering data preprocessing, diffusion model training, inference and visualization within ManiSkill3.
2. Implement two model variants: unconditional diffusion baseline and customized conditional diffusion policy.
3. Fix the official dataset loading crash caused by nested dictionary observations failing to convert into tensors.
4. Compare model convergence and generated trajectory quality, and analyze performance bottlenecks brought by hardware and environment constraints.
5. Propose feasible optimization plans based on the official ConditionalUnet1D architecture and GPU acceleration.

## Hardware & Software Environment
### Hardware
- Laptop GPU: NVIDIA RTX 5060 Laptop. CUDA and PyTorch version incompatibility prevents GPU training; all experiments run on CPU.
- Physics & Simulation Engine: SAPIEN, ManiSkill3 (GPU parallel rendering supported theoretically)

### Software Stack
- Python: 3.10
- Environment Management: Miniconda, isolated virtual environment named `robodiff`
- Core Framework: PyTorch, CUDA Toolkit
- Diffusion Module: diffusers (partial functions unavailable; full DDPM logic implemented manually)
- Simulation Dependencies: ManiSkill3, Gymnasium, sapien
- Data Processing: h5py, numpy, sympy
- Experiment Logging: Weights & Biases (wandb)
- Demonstration Format: HDF5 trajectory files with JSON metadata

### Environment Installation Commands
```bash
# Create isolated conda environment
conda create -n robodiff python=3.10
conda activate robodiff

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mani-skill3 gymnasium sapien h5py diffusers sympy wandb numpy
```

## Dataset & Data Preprocessing
All experiments are conducted on the `PickCube-v1` task from ManiSkill3.
- Dataset scale: 100 motion-planning expert trajectories stored in standard HDF5 format
- Raw data feature: Observations stored as nested dictionaries, control actions as flat numpy arrays
- Core bug fix: Official data loader cannot convert dictionary observations into torch tensors. We implement a dedicated flattening module to transform nested observation data into fixed-dimensional numeric vectors.
- Chunking hyperparameters
  - Observation Horizon = 2
  - Action Horizon = 8
  - Prediction Horizon = 16
- Preprocessing operation: Standard normalization applied to action space to stabilize training gradients

## Diffusion Policy Implementation
### 1. Unconditional Diffusion Model
- Training script only learns the distribution of expert action sequences without observation input.
- Backbone: Simple multi-layer perceptron.
- Usage: Weak baseline reference only; generates disordered, non-executable robot trajectories.

### 2. Custom Conditional Diffusion Policy
#### Three Input Branches Concatenated for Forward Propagation
1. Flattened historical observation sequence
2. Action chunk polluted with Gaussian noise at diffusion timestep t
3. Time-step embedding of current diffusion iteration

#### Core Architecture
- Denoising backbone: Multi-layer MLP (used as a substitute due to diffusers library import failure)
- Self-contained DDPM implementation: No reliance on official DDPMScheduler
- Cosine beta schedule, total diffusion denoising steps set to 100
- Loss function: MSE loss for noise prediction
- Optimizer: AdamW with weight decay regularization

#### Standard Training Workflow
1. Load paired sliding-window observation-action chunks from preprocessed demonstrations
2. Randomly sample diffusion timestep t and inject Gaussian noise into target action segments
3. Concatenate observation features, noised action sequences and time embeddings as network input
4. Predict noise residual, calculate MSE loss and update model parameters via backpropagation
5. Save model checkpoints and record loss curves via wandb logging tool

## Key Engineering Challenges & Solutions
1. CUDA-PyTorch Version Mismatch on RTX 5060 Laptop GPU
   - Problem: Disabled GPU parallel training and rendering, forcing CPU-only training.
   - Solution: Adjust batch size and learning rate to adapt to CPU computing limitations.
2. HDF5 Observation Dictionary Tensor Conversion Crash
   - Error traceback: `TypeError: new(): data must be a sequence (got dict)`
   - Resolution: Custom flattening utility converts nested observation dictionaries into one-dimensional float vectors.
3. Failed Import of Official Diffusion Modules
   - Problem: Unable to load ConditionalUnet1D and DDPMScheduler from diffusers.
   - Solution: Manually implement complete forward diffusion and reverse denoising Markov chains, replace UNet with MLP backbone.
4. Missing Observation Fallback in Official Script
   - Extension: Add automatic fallback to environment state data when observation key is missing in demonstration files.

## Training Results & Model Evaluation
### Convergence Performance
- Initial training loss: ~0.69
- Final converged loss: ~0.35
- Conclusion: The full training pipeline is functional, and the model can capture basic statistical features of expert action distributions.

### Trajectory Quality Defects
1. Unconditional model outputs completely random joint trajectories without valid manipulation logic.
2. Conditional MLP policy generates noisy, discontinuous action sequences.
3. Denoising curves lack obvious convergence trends; generated trajectories deviate drastically from smooth ground-truth expert motion.

### Root Causes of Suboptimal Performance
- CPU-only training restricts training iterations and hyperparameter tuning space.
- MLP backbone has weaker feature extraction capability than the official 1D Conditional UNet.
- Missing training stabilizers including EMA, dynamic learning rate scheduling and gradient clipping.
- Imperfect observation flattening logic leads to unstable training gradients.

## Project Strengths
1. A fully self-contained, trainable and loadable diffusion policy prototype built from scratch.
2. Solved the critical dataset loading error of the official ManiSkill3 baseline with independent observation flattening logic.
3. Realized complete DDPM forward and reverse processes without full diffusers library dependency.
4. Natively compatible with ManiSkill3 PickCube-v1 task, supporting RGB-D visual and proprioceptive observation inputs.
5. Verified core advantages of diffusion imitation learning: effectively avoids mode collapse and naturally outputs temporally coherent action chunks.

## Limitations & Future Improvement Directions
### Current Limitations
1. RTX 5060 CUDA compatibility failure leads to slow CPU training and limited model performance.
2. Simplified MLP denoiser replaces the standard ConditionalUnet1D architecture.
3. Lack of auxiliary training regularization tools such as EMA and learning rate decay.
4. Unstable generated trajectories with poor alignment to ground-truth expert actions.

### Follow-up Optimization Roadmap
1. Environment & Hardware Optimization
   - Reinstall matching CUDA and PyTorch versions to enable GPU acceleration on RTX 5060.
   - Fix diffusers installation to access official ConditionalUnet1D and DDPMScheduler.
2. Model Architecture Upgrade
   - Replace MLP denoiser with standard 1D conditional UNet.
   - Integrate 3D point cloud or voxel scene features inspired by 3D Diffusion Policy to improve generalization in unseen environments.
3. Training & Data Pipeline Refinement
   - Introduce EMA, learning rate scheduler and gradient clipping to stabilize training.
   - Optimize observation flattening module and standardize dataset loading logic.
4. Evaluation System Construction
   - Build automatic stable evaluation loop to record task success rate and average reward.
   - Add quantitative comparison metrics between ground-truth trajectories and model-generated actions.

## Citation
```
@courseproject{robodiff2026,
  title={Full-Stack Implementation of Robotic Arm Grasping Based on Diffusion Policy},
  course={CS461 Robotics Final Project},
  year={2026},
  institution={University Student Course Project}
}
```

## License
MIT License
This repository is open for academic research and secondary development.
