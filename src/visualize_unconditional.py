import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.common import flatten_state_dict
import os
import warnings

# 设置环境变量解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 如需显示窗口，请不要强制使用 EGL 离屏渲染
#warnings.filterwarnings("ignore")

# ====================== 配置 ======================
TASK = "PickCube-v1"
MODEL_PATH = "diffusion_policy_final.pt"
DEVICE = torch.device("cpu")

# ====================== 模型定义（与训练脚本一致） ======================
class UnconditionalDiffusionPolicy(nn.Module):
    def __init__(self, act_dim, hidden_dim=1024):
        super().__init__()
        self.time_embedding = nn.Embedding(200, hidden_dim)  # DIFF_STEPS = 200
        self.decoder = nn.Sequential(
            nn.Linear(act_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, noisy_action, timestep):
        time_feat = self.time_embedding(timestep)
        feat = torch.cat([noisy_action, time_feat], dim=-1)
        return self.decoder(feat)

# ====================== 推理函数 ======================
def sample_from_model(model, act_dim, act_mean, act_std, device):
    """从训练好的模型中采样动作"""
    model.eval()
    with torch.no_grad():
        # 从纯噪声开始
        x = torch.randn(1, act_dim, device=device)

        # 逆扩散过程
        for t in reversed(range(200)):  # DIFF_STEPS = 200
            timestep = torch.tensor([t], device=device)
            pred_noise = model(x, timestep)

            # 简化的逆扩散步骤（DDPM公式）
            beta = torch.linspace(1e-4, 2e-2, 200, device=device)[t]
            alpha = 1.0 - beta
            alpha_hat = torch.cumprod(torch.linspace(1e-4, 2e-2, 200, device=device), dim=0)[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * pred_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise

        # 反归一化
        x = x * act_std + act_mean
        return x.squeeze(0).cpu().numpy()

# ====================== 主函数 ======================
def main():
    print("加载模型...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    act_dim = ckpt["act_dim"]
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]

    model = UnconditionalDiffusionPolicy(act_dim).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print("模型加载完成")

    # 创建环境
    env = gym.make(TASK, obs_mode="none", control_mode="pd_joint_pos", render_mode="human")
    print(f"环境创建成功: task={TASK}, control_mode=pd_joint_pos")

    # 运行几个episode
    for episode in range(3):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        print(f"\nEpisode {episode + 1}")
        while not (terminated or truncated) and step < 50:
            # 使用扩散模型采样动作
            action = sample_from_model(model, act_dim, act_mean, act_std, DEVICE)
            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(env, 'render'):
                try:
                    env.render()
                except Exception:
                    pass
            total_reward += reward
            step += 1

        print(f"Episode {episode + 1} finished. Total reward: {total_reward.item():.2f}, Steps: {step}")

    env.close()

if __name__ == "__main__":
    main()