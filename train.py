"""
训练脚本：使用合成数据训练击球决策神经网络
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from environment import (
    generate_random_state,
    compute_optimal_action,
    normalize_state,
    STATE_DIM,
)
from model import build_model


# ── 默认超参数 ──────────────────────────────────────────────────────────────────
DEFAULT_NUM_SAMPLES = 50_000
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
MODEL_SAVE_PATH = "strike_model.pth"


def generate_dataset(num_samples, seed=42):
    """
    生成监督学习训练数据集。

    参数:
        num_samples (int): 样本数量
        seed        (int): 随机种子，保证可复现
    返回:
        (np.ndarray, np.ndarray): 状态矩阵 (N, STATE_DIM)，动作矩阵 (N, ACTION_DIM)
    """
    rng = np.random.default_rng(seed)
    states = np.zeros((num_samples, STATE_DIM), dtype=np.float32)
    actions = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        state = generate_random_state(rng)
        states[i] = normalize_state(state.to_vector())
        actions[i] = compute_optimal_action(state)

    return states, actions


def train(
    num_samples=DEFAULT_NUM_SAMPLES,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    save_path=MODEL_SAVE_PATH,
    hidden_sizes=(128, 128),
    verbose=True,
):
    """
    训练击球决策网络并保存模型权重。

    参数:
        num_samples  (int)  : 训练样本数
        epochs       (int)  : 训练轮数
        batch_size   (int)  : 批大小
        lr           (float): 学习率
        save_path    (str)  : 模型保存路径
        hidden_sizes (tuple): 隐藏层尺寸
        verbose      (bool) : 是否打印训练进度
    返回:
        list[float]: 每轮的平均训练损失
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"使用设备: {device}")
        print(f"生成 {num_samples} 条训练样本...")

    states, actions = generate_dataset(num_samples)

    x_tensor = torch.from_numpy(states)
    y_tensor = torch.from_numpy(actions)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_model(hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if verbose:
        print(f"开始训练，共 {epochs} 轮...")

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch [{epoch:>3}/{epochs}]  Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    if verbose:
        print(f"模型已保存至: {os.path.abspath(save_path)}")

    return loss_history


if __name__ == "__main__":
    train()
