"""
击球决策推理模块
加载训练好的神经网络，对给定游戏状态输出击球动作
"""

import os

import numpy as np
import torch

from environment import GameState, StrikeAction, normalize_state
from model import build_model

DEFAULT_MODEL_PATH = "strike_model.pth"


class StrikeDecisionMaker:
    """
    封装训练好的神经网络，提供便捷的击球决策接口。

    用法示例::

        dm = StrikeDecisionMaker()
        state = GameState(ball_pos=[2, 1, 2], ball_vel=[-3, 4, -1], robot_pos=[-4, 0, 1])
        action = dm.decide(state)
        print(action)
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, hidden_sizes=(128, 128), device=None):
        """
        初始化决策器。

        参数:
            model_path   (str)  : 训练好的模型权重路径
            hidden_sizes (tuple): 与训练时相同的隐藏层尺寸
            device       (str)  : 推理设备（默认自动选择）
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = build_model(hidden_sizes=hidden_sizes).to(self.device)

        if os.path.isfile(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(
                f"找不到模型文件: {model_path}\n"
                "请先运行 train.py 进行训练。"
            )

        self.model.eval()

    def decide(self, state):
        """
        根据当前游戏状态输出击球决策。

        参数:
            state (GameState): 当前游戏状态
        返回:
            StrikeAction: 推荐的击球动作
        """
        state_vec = normalize_state(state.to_vector())
        x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_tensor = self.model(x)

        action_vec = action_tensor.squeeze(0).cpu().numpy()
        return StrikeAction(
            horizontal_angle=action_vec[0],
            vertical_angle=action_vec[1],
            force=action_vec[2],
        )

    def decide_batch(self, states):
        """
        批量推理，一次处理多个状态。

        参数:
            states (list[GameState]): 游戏状态列表
        返回:
            list[StrikeAction]: 对应的击球动作列表
        """
        state_vecs = np.stack([normalize_state(s.to_vector()) for s in states])
        x = torch.tensor(state_vecs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_tensors = self.model(x)

        actions = []
        for av in action_tensors.cpu().numpy():
            actions.append(StrikeAction(
                horizontal_angle=av[0],
                vertical_angle=av[1],
                force=av[2],
            ))
        return actions


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    dm = StrikeDecisionMaker(model_path=model_path)

    test_state = GameState(
        ball_pos=[2.0, 1.0, 2.5],
        ball_vel=[-3.0, 4.0, -1.5],
        robot_pos=[-4.0, 0.5, 1.0],
    )

    action = dm.decide(test_state)
    print("游戏状态:", test_state)
    print("击球决策:", action)
    print("对应速度:", action.to_velocity())
