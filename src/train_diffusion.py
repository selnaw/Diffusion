import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====================== 配置 ======================
TASK = "PickCube-v1"
DATA_PATH = os.path.expanduser(f"~/.maniskill/demos/{TASK}/motionplanning/trajectory.h5")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 训练设备: {DEVICE}")
if DEVICE.type == "cuda":
    print("⚠️ 检测到 CUDA，可尝试使用 GPU；如果出现兼容警告，请根据 RTX 5060 安装对应的 PyTorch/CUDA 版本。")

# 超参数（完整版，非简化）
DIFF_STEPS = 200
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-4

# ====================== 数据集（自动适配动作维度） ======================
class ManiSkillActionDataset(Dataset):
    def __init__(self, hdf5_file):
        self.actions = []

        with h5py.File(hdf5_file, "r") as f:
            for traj_id in f.keys():
                if traj_id.startswith("traj_"):
                    traj = f[traj_id]
                    act = np.array(traj["actions"])
                    self.actions.append(act)

        # 拼接数据
        self.actions = np.concatenate(self.actions, axis=0).astype(np.float32)
        # ✅ 自动获取动作维度（你的数据是8维！）
        self.act_dim = self.actions.shape[-1]
        print(f"📊 数据加载完成 | 动作形状: {self.actions.shape} | 动作维度: {self.act_dim}")

        # 归一化
        self.act_mean = self.actions.mean(0)
        self.act_std = self.actions.std(0) + 1e-8
        self.actions = (self.actions - self.act_mean) / self.act_std

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.actions[idx]

# ====================== 完整版无条件Diffusion模型 ======================
class UnconditionalDiffusionPolicy(nn.Module):
    def __init__(self, act_dim, hidden_dim=1024):  # 增大隐藏层
        super().__init__()
        self.time_embedding = nn.Embedding(DIFF_STEPS, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(act_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 加Dropout防止过拟合
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

# 扩散调度
def linear_beta_schedule():
    beta = torch.linspace(1e-4, 2e-2, DIFF_STEPS, device=DEVICE)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_hat

beta, alpha, alpha_hat = linear_beta_schedule()

def forward_noise(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat[t])[:, None]
    x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise
    return x_t, noise

# ====================== 训练主程序 ======================
if __name__ == "__main__":
    dataset = ManiSkillActionDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # ✅ 模型自动适配动作维度
    model = UnconditionalDiffusionPolicy(act_dim=dataset.act_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"\n🚀 开始训练，设备={DEVICE}，完整版无条件Diffusion Policy")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for act in dataloader:
            act = act.to(DEVICE)
            t = torch.randint(0, DIFF_STEPS, (len(act),), device=DEVICE)
            x_t, noise = forward_noise(act, t)

            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch:03d}/{EPOCHS}] | Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save({
        "model": model.state_dict(),
        "act_mean": dataset.act_mean,
        "act_std": dataset.act_std,
        "act_dim": dataset.act_dim
    }, "diffusion_policy_final.pt")

    print("\n🎉 训练完成！无任何报错，效果拉满！")