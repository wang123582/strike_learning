"""
排球机器人击球决策神经网络模型
"""

import torch
import torch.nn as nn

from environment import STATE_DIM, ACTION_DIM, ACTION_LOW, ACTION_HIGH


class StrikeNet(nn.Module):
    """
    全连接前馈神经网络，用于击球决策。

    输入：归一化后的游戏状态向量（维度 STATE_DIM）
    输出：击球动作向量（维度 ACTION_DIM），各分量已缩放到真实范围
    """

    def __init__(self, hidden_sizes=(128, 128)):
        """
        初始化网络。

        参数:
            hidden_sizes (tuple): 各隐藏层的神经元数量
        """
        super().__init__()

        layers = []
        in_size = STATE_DIM
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, ACTION_DIM))
        layers.append(nn.Tanh())  # 输出范围 (-1, 1)，再映射到动作范围

        self.net = nn.Sequential(*layers)

        # 将动作范围注册为缓冲区（跟随模型保存/加载）
        low = torch.tensor(ACTION_LOW, dtype=torch.float32)
        high = torch.tensor(ACTION_HIGH, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (Tensor): 形状 (batch, STATE_DIM) 的归一化状态张量
        返回:
            Tensor: 形状 (batch, ACTION_DIM) 的动作张量（真实范围）
        """
        raw = self.net(x)  # (-1, 1)
        # 线性映射到 [action_low, action_high]
        action = self.action_low + (raw + 1.0) * 0.5 * (self.action_high - self.action_low)
        return action


def build_model(hidden_sizes=(128, 128)):
    """
    工厂函数：构建并初始化 StrikeNet。

    参数:
        hidden_sizes (tuple): 各隐藏层的神经元数量
    返回:
        StrikeNet
    """
    model = StrikeNet(hidden_sizes=hidden_sizes)
    # Xavier 初始化
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model
