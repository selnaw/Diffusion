import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from mani_skill.utils.common import flatten_state_dict

# 尝试导入 ManiSkill 官方 diffusion_policy utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_UTILS_DIR = os.path.join(
    ROOT_DIR,
    "ManiSkill",
    "examples",
    "baselines",
    "diffusion_policy",
)
if os.path.isdir(DIFFUSION_UTILS_DIR):
    sys.path.insert(0, DIFFUSION_UTILS_DIR)
try:
    from diffusion_policy.utils import load_demo_dataset
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    from diffusers.optimization import get_scheduler
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
except Exception as e:
    print(f"⚠️ 无法导入官方 diffusion_policy 组件: {e}")
    print("将使用简化版本的扩散策略。")
    load_demo_dataset = None
    DDPMScheduler = None
    EMAModel = None
    get_scheduler = None
    ConditionalUnet1D = None


@dataclass
class Args:
    task: str = "PickCube-v1"
    demo_path: str = os.path.expanduser("~/.maniskill/demos/PickCube-v1")
    save_path: str = "cond_best_optimized.pt"
    device: str = "cpu"  # RTX 5060 不支持当前 PyTorch CUDA，使用 CPU
    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-4
    diff_steps: int = 100
    hidden: int = 256
    num_workers: int = 0
    use_official_loader: bool = True
    # Diffusion Policy specific
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    control_mode: str = "pd_joint_pos"


class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, demo_path, device, obs_horizon, pred_horizon, num_traj=None):
        demo_path = os.path.expanduser(demo_path)
        if load_demo_dataset is not None:
            trajectories = load_demo_dataset(demo_path, num_traj=num_traj, concat=False)
        else:
            # Fallback to h5py loading
            trajectories = self._load_with_h5py(demo_path, num_traj)

        # Convert to tensors
        for k, v in trajectories.items():
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    # Flatten state dict
                    v_flat = []
                    for key in sorted(v[i].keys()):
                        val = v[i][key]
                        if isinstance(val, list):
                            val = np.array(val)
                        v_flat.append(torch.Tensor(val).flatten())
                    trajectories[k][i] = torch.cat(v_flat).to(device)
                else:
                    trajectories[k][i] = torch.Tensor(v[i]).to(device)

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx].shape[0] == L + 1
            total_transitions += L

            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories

    def _load_with_h5py(self, demo_path, num_traj=None):
        import h5py
        trajectories = {'observations': [], 'actions': []}
        with h5py.File(demo_path, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            traj_keys.sort(key=lambda x: int(x.split("_")[-1]))
            if num_traj is not None:
                traj_keys = traj_keys[:num_traj]
            for traj_key in traj_keys:
                traj = f[traj_key]
                obs = self._load_obs_from_traj(traj)
                act = np.array(traj["actions"])
                trajectories['observations'].append(obs)
                trajectories['actions'].append(act)
        return trajectories

    def _load_obs_from_traj(self, traj):
        import h5py
        if "env_states" in traj:
            env_states = traj["env_states"]
            step_state_dicts = []
            sample_group = env_states["actors"] if "actors" in env_states else env_states["articulations"]
            sample_key = sorted(sample_group.keys())[0]
            num_steps = np.array(sample_group[sample_key]).shape[0]

            for t in range(num_steps):
                state_dict = {"actors": {}, "articulations": {}}
                for group_name in ["actors", "articulations"]:
                    if group_name not in env_states:
                        continue
                    group = env_states[group_name]
                    for entity_name in sorted(group.keys()):
                        arr = np.array(group[entity_name])
                        state_dict[group_name][entity_name] = arr[t]
                step_state_dicts.append(state_dict)

            flattened = [flatten_state_dict(state, use_torch=True, device='cpu').cpu().numpy() for state in step_state_dicts]
            return np.stack(flattened, axis=0)
        else:
            raise RuntimeError("Only env_states loading supported for official-style dataset")

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_seq = self.trajectories['observations'][traj_idx][max(0, start):start+self.obs_horizon]
        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0:
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            pad_action = act_seq[-1].repeat(end-L, 1)
            act_seq = torch.cat([act_seq, pad_action], dim=0)
        assert obs_seq.shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {
            'observations': obs_seq,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)


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

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]
        obs_cond = obs_seq.flatten(start_dim=1)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps if hasattr(self, 'noise_scheduler') else self.num_diffusion_iters, (B,), device=obs_seq.device).long()

        if hasattr(self, 'noise_scheduler'):
            noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        else:
            # Fallback - ensure all tensors are on the same device
            device = obs_seq.device
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[timesteps].to(device))[:, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[timesteps].to(device))[:, None, None]
            noisy_action_seq = sqrt_alpha_hat * action_seq + sqrt_one_minus_alpha_hat * noise

        if ConditionalUnet1D is not None and isinstance(self.noise_pred_net, ConditionalUnet1D):
            noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        else:
            # MLP fallback
            time_feat = self.time_emb(timesteps)
            x = torch.cat([noisy_action_seq.flatten(start_dim=1), obs_cond, time_feat], dim=-1)
            noise_pred_flat = self.noise_pred_net(x)
            noise_pred = noise_pred_flat.view(B, self.pred_horizon, self.act_dim)

        return nn.functional.mse_loss(noise_pred, noise)

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


def cosine_beta_schedule(num_steps, device):
    s = 0.008
    betas = []
    for i in range(num_steps):
        t1 = (i / num_steps + s) / (1 + s) * np.pi / 2
        t2 = ((i + 1) / num_steps + s) / (1 + s) * np.pi / 2
        beta = 1 - np.cos(t2) ** 2 / np.cos(t1) ** 2
        betas.append(np.clip(beta, 0.0001, 0.02))
    return torch.tensor(betas, dtype=torch.float32, device=device)


def forward_noise(x_start, timesteps, alpha_hat):
    noise = torch.randn_like(x_start)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[timesteps])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat[timesteps])[:, None]
    x_noisy = sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise
    return x_noisy, noise


def resolve_demo_path(raw_path):
    demo_path = os.path.expanduser(raw_path)
    demo_path = os.path.expandvars(demo_path)
    demo_path = os.path.normpath(demo_path)

    if os.path.isfile(demo_path):
        return demo_path

    if os.path.isdir(demo_path):
        candidates = [
            os.path.join(demo_path, "trajectory.h5"),
            os.path.join(demo_path, "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
            os.path.join(demo_path, "motionplanning", "trajectory.h5"),
            os.path.join(demo_path, "motionplanning", "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

    parent = os.path.dirname(demo_path)
    candidates = [
        demo_path,
        os.path.join(parent, "trajectory.h5"),
        os.path.join(parent, "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
        os.path.join(parent, "motionplanning", "trajectory.h5"),
        os.path.join(parent, "motionplanning", "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.normpath(path)

    raise FileNotFoundError(
        "演示数据文件不存在: {}\n请检查路径是否正确，或使用以下文件名之一：trajectory.h5, trajectory.state.pd_ee_delta_pos.physx_cpu.h5".format(demo_path)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated ManiSkill diffusion policy training")
    parser.add_argument("--task", default="PickCube-v1", help="ManiSkill 环境名")
    parser.add_argument(
        "--demo-path",
        default=os.path.expanduser("~/.maniskill/demos/PickCube-v1"),
        help="演示数据路径，可以是目录或 h5 文件",
    )
    parser.add_argument(
        "--save-path",
        default="cond_best_optimized.pt",
        help="保存模型的文件名",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--diff-steps", type=int, default=100, help="扩散步数")
    parser.add_argument("--hidden", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--use-official-loader",
        action="store_true",
        help="优先使用 ManiSkill 官方 load_demo_dataset 加载数据",
    )
    # Diffusion Policy specific
    parser.add_argument("--obs-horizon", type=int, default=2, help="观察序列长度")
    parser.add_argument("--act-horizon", type=int, default=8, help="动作序列长度")
    parser.add_argument("--pred-horizon", type=int, default=16, help="预测序列长度")
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64, help="扩散时间嵌入维度")
    parser.add_argument("--unet-dims", type=int, nargs='+', default=[64, 128, 256], help="U-Net 维度")
    parser.add_argument("--n-groups", type=int, default=8, help="U-Net 组数")
    parser.add_argument("--control-mode", default="pd_joint_pos", help="控制模式")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    demo_path = resolve_demo_path(args.demo_path)
    print(f"🔎 使用展开后的 demo_path: {demo_path}")

    # Convert args to dataclass for compatibility
    args_dict = vars(args)
    training_args = Args(**args_dict)

    dataset = SmallDemoDataset_DiffusionPolicy(demo_path, device, training_args.obs_horizon, training_args.pred_horizon)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Get obs_dim from dataset
    sample = dataset[0]
    obs_dim = sample['observations'].shape[-1]

    model = Agent(obs_dim, 8, training_args, device).to(device)  # act_dim=8 for PickCube
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    if get_scheduler is not None:
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=args.epochs * len(dataloader),
        )
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if EMAModel is not None:
        ema = EMAModel(parameters=model.parameters(), power=0.75)
        ema_model = Agent(obs_dim, 8, training_args, device).to(device)
    else:
        ema = None
        ema_model = None

    criterion = nn.MSELoss()

    print("\n🚀 开始训练")
    print(f"  task={args.task}")
    print(f"  demo_path={args.demo_path}")
    print(f"  save_path={args.save_path}")
    print(f"  obs_dim={obs_dim}, act_dim=8")
    print(f"  device={device}, batch_size={args.batch_size}, epochs={args.epochs}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for data_batch in dataloader:
            obs_seq = data_batch["observations"].to(device)
            action_seq = data_batch["actions"].to(device)

            loss = model.compute_loss(obs_seq, action_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if get_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.step(model.parameters())

            total_loss += float(loss.item()) * len(obs_seq)

        if get_scheduler is None:
            lr_scheduler.step()
        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs}] loss={avg_loss:.6f}")

    print("\n✅ 训练完成，开始保存模型...")
    save_dict = {
        "obs_dim": obs_dim,
        "act_dim": 8,
        "model": model.state_dict(),
        "args": args_dict,
    }
    if ema_model is not None:
        ema.copy_to(ema_model.parameters())
        save_dict["ema_model"] = ema_model.state_dict()
    torch.save(save_dict, args.save_path)
    print(f"✅ 模型已保存为 {args.save_path}")
    print(f"现在可以使用 visualize_final.py --model-path {args.save_path} 进行可视化。")


if __name__ == "__main__":
    main()
