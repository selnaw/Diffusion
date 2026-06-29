import argparse
import mani_skill
import mani_skill.envs
from mani_skill.utils.common import flatten_state_dict
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import warnings
import os
from dataclasses import dataclass, field
from typing import List

# 强制使用软件渲染（避免 OpenGL 问题）
os.environ['PYOPENGL_PLATFORM'] = 'egl'

warnings.filterwarnings("ignore")

# Try to import official components
try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
except Exception:
    DDPMScheduler = None
    ConditionalUnet1D = None


@dataclass
class Args:
    task: str = "PickCube-v1"
    demo_path: str = ""
    save_path: str = "cond_best_optimized.pt"
    device: str = "cpu"
    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-4
    diff_steps: int = 100
    hidden: int = 256
    num_workers: int = 0
    use_official_loader: bool = True
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    control_mode: str = "pd_joint_pos"


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, args, device):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.act_dim = act_dim

        if ConditionalUnet1D is not None:
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.act_dim,
                global_cond_dim=obs_dim * self.obs_horizon,
                diffusion_step_embed_dim=args.diffusion_step_embed_dim,
                down_dims=args.unet_dims,
                n_groups=args.n_groups,
            )
        else:
            # Fallback to simple MLP
            # Input: noisy_action_seq (B, pred_horizon, act_dim) + obs_cond (B, obs_dim*obs_horizon) + time_emb (B, diffusion_step_embed_dim)
            input_dim = self.pred_horizon * self.act_dim + obs_dim * self.obs_horizon + args.diffusion_step_embed_dim
            self.noise_pred_net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, self.pred_horizon * self.act_dim),
            )
            self.time_emb = nn.Embedding(args.diff_steps, args.diffusion_step_embed_dim)

        self.num_diffusion_iters = args.diff_steps
        if DDPMScheduler is not None:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        else:
            # Simple fallback scheduler
            self.beta = torch.linspace(0.0001, 0.02, self.num_diffusion_iters, device=device)
            self.alpha = 1.0 - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def get_action(self, obs_seq):
        B = obs_seq.shape[0]
        obs_cond = obs_seq.flatten(start_dim=1)

        with torch.no_grad():
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

            if hasattr(self, 'noise_scheduler'):
                for k in self.noise_scheduler.timesteps:
                    if ConditionalUnet1D is not None and isinstance(self.noise_pred_net, ConditionalUnet1D):
                        noise_pred = self.noise_pred_net(
                            sample=noisy_action_seq,
                            timestep=k,
                            global_cond=obs_cond,
                        )
                    else:
                        time_feat = self.time_emb(k.repeat(B))
                        x = torch.cat([noisy_action_seq.flatten(start_dim=1), obs_cond, time_feat], dim=-1)
                        noise_pred_flat = self.noise_pred_net(x)
                        noise_pred = noise_pred_flat.view(B, self.pred_horizon, self.act_dim)

                    noisy_action_seq = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=noisy_action_seq,
                    ).prev_sample
            else:
                # Simple reverse diffusion fallback
                for t in reversed(range(self.num_diffusion_iters)):
                    if ConditionalUnet1D is not None and isinstance(self.noise_pred_net, ConditionalUnet1D):
                        noise_pred = self.noise_pred_net(
                            sample=noisy_action_seq,
                            timestep=torch.tensor([t], device=obs_seq.device).repeat(B),
                            global_cond=obs_cond,
                        )
                    else:
                        time_feat = self.time_emb(torch.tensor([t], device=obs_seq.device).repeat(B))
                        x = torch.cat([noisy_action_seq.flatten(start_dim=1), obs_cond, time_feat], dim=-1)
                        noise_pred_flat = self.noise_pred_net(x)
                        noise_pred = noise_pred_flat.view(B, self.pred_horizon, self.act_dim)

                    alpha_t = self.alpha[t].to(obs_seq.device)
                    alpha_hat_t = self.alpha_hat[t].to(obs_seq.device)
                    beta_t = self.beta[t].to(obs_seq.device)
                    noisy_action_seq = (noisy_action_seq - (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_t)
                    if t > 0:
                        noise = torch.randn_like(noisy_action_seq)
                        noisy_action_seq += torch.sqrt(beta_t) * noise

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained diffusion policy")
    parser.add_argument("--task", default="PickCube-v1", help="ManiSkill 环境名")
    parser.add_argument(
        "--model-path",
        default="cond_best_optimized.pt",
        help="训练完成后保存的模型文件路径",
    )
    parser.add_argument("--obs-mode", default="none", help="环境 obs_mode")
    parser.add_argument("--control-mode", default="pd_joint_pos", help="环境 control_mode")
    parser.add_argument("--reward-mode", default="dense", help="环境 reward_mode")
    parser.add_argument("--episodes", type=int, default=3, help="演示多少集")
    return parser.parse_args()


def build_obs_seq(env, obs, obs_dim, obs_horizon):
    """Build observation sequence for diffusion policy"""
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().detach().numpy()
    obs = np.asarray(obs).reshape(-1)
    if obs.shape[0] != obs_dim:
        if hasattr(env, "base_env") and hasattr(env.base_env, "get_state_dict"):
            state_dict = env.base_env.get_state_dict()
            obs = flatten_state_dict(state_dict, use_torch=True, device='cpu').cpu().numpy()
            obs = np.asarray(obs).reshape(-1)
        else:
            raise RuntimeError(
                f"当前环境 observation 维度 {obs.shape[0]} 与模型期望 {obs_dim} 不匹配，且无法使用 state_dict 进行重建。"
            )
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cpu')[None]  # (1, obs_dim)
    # Repeat for obs_horizon
    obs_seq = obs_tensor.repeat(obs_horizon, 1).unsqueeze(0)  # (1, obs_horizon, obs_dim)
    return obs_seq


def main():
    args = parse_args()
    DEVICE = torch.device("cpu")
    TASK = args.task
    MODEL_PATH = args.model_path

    # Load model
    print("加载模型...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    model_args = ckpt.get("args", {})
    training_args = Args(**model_args) if model_args else Args()

    model = Agent(obs_dim, act_dim, training_args, DEVICE).to(DEVICE)
    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    print("模型加载完成")

    # Create environment
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "human",
    }
    env = gym.make(TASK, **env_kwargs)
    print(f"环境创建成功: task={TASK}, obs_mode={args.obs_mode}, control_mode={args.control_mode}, reward_mode={args.reward_mode}")

    for episode in range(args.episodes):
        print(f"\nEpisode {episode+1}")
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        while not done and step < 300:
            obs_seq = build_obs_seq(env, obs, obs_dim, training_args.obs_horizon).to(DEVICE)
            action_seq = model.get_action(obs_seq)  # (1, act_horizon, act_dim)
            action = action_seq[0, 0].cpu().numpy()  # Take first action

            obs, reward, terminated, truncated, _ = env.step(action)
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
            total_reward += reward
            done = terminated or truncated
            step += 1
            env.render()

        print(f"Episode {episode+1} finished. Total reward: {total_reward:.2f}, Steps: {step}")

    env.close()


if __name__ == "__main__":
    main()