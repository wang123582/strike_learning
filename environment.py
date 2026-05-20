"""
排球机器人击球环境模块
定义游戏状态空间、动作空间及相关物理参数
"""

import numpy as np


# 场地尺寸（米）
COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_HEIGHT = 2.43
COURT_HALF = COURT_LENGTH / 2.0

# 物理参数
GRAVITY = 9.8          # 重力加速度 (m/s^2)
BALL_RADIUS = 0.105    # 排球半径 (m)

# 状态维度
STATE_DIM = 9  # [ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, robot_x, robot_y, robot_z]

# 动作维度
ACTION_DIM = 3  # [horizontal_angle, vertical_angle, force]

# 动作范围
ACTION_LOW = np.array([-np.pi / 2, 0.0, 5.0], dtype=np.float32)   # 最小值
ACTION_HIGH = np.array([np.pi / 2, np.pi / 3, 25.0], dtype=np.float32)  # 最大值


class GameState:
    """表示排球比赛的瞬时状态"""

    def __init__(self, ball_pos, ball_vel, robot_pos):
        """
        初始化游戏状态。

        参数:
            ball_pos (array-like): 球的三维坐标 [x, y, z]（米）
            ball_vel (array-like): 球的三维速度 [vx, vy, vz]（米/秒）
            robot_pos (array-like): 机器人三维坐标 [x, y, z]（米）
        """
        self.ball_pos = np.array(ball_pos, dtype=np.float32)
        self.ball_vel = np.array(ball_vel, dtype=np.float32)
        self.robot_pos = np.array(robot_pos, dtype=np.float32)

    def to_vector(self):
        """将状态转换为一维特征向量"""
        return np.concatenate([self.ball_pos, self.ball_vel, self.robot_pos])

    def __repr__(self):
        return (
            f"GameState(ball_pos={self.ball_pos}, "
            f"ball_vel={self.ball_vel}, "
            f"robot_pos={self.robot_pos})"
        )


class StrikeAction:
    """表示一次击球动作"""

    def __init__(self, horizontal_angle, vertical_angle, force):
        """
        初始化击球动作。

        参数:
            horizontal_angle (float): 水平击球角度（弧度），0 表示正前方
            vertical_angle   (float): 垂直击球角度（弧度），0 表示水平
            force            (float): 击球力度（米/秒）
        """
        self.horizontal_angle = float(horizontal_angle)
        self.vertical_angle = float(vertical_angle)
        self.force = float(force)

    def to_vector(self):
        """将动作转换为一维向量"""
        return np.array([self.horizontal_angle, self.vertical_angle, self.force],
                        dtype=np.float32)

    def to_velocity(self):
        """将动作转换为球的初始速度向量 [vx, vy, vz]"""
        vx = self.force * np.cos(self.vertical_angle) * np.sin(self.horizontal_angle)
        vy = self.force * np.cos(self.vertical_angle) * np.cos(self.horizontal_angle)
        vz = self.force * np.sin(self.vertical_angle)
        return np.array([vx, vy, vz], dtype=np.float32)

    def __repr__(self):
        return (
            f"StrikeAction(h_angle={self.horizontal_angle:.3f} rad, "
            f"v_angle={self.vertical_angle:.3f} rad, "
            f"force={self.force:.1f} m/s)"
        )


def normalize_state(state_vec, low=None, high=None):
    """
    将状态向量归一化到 [-1, 1]。

    参数:
        state_vec (np.ndarray): 原始状态向量
        low  (np.ndarray): 各维度最小值（默认按场地参数推导）
        high (np.ndarray): 各维度最大值
    返回:
        np.ndarray: 归一化后的状态向量
    """
    if low is None:
        low = np.array(
            [-COURT_HALF, -COURT_WIDTH / 2, 0.0,
             -30.0, -30.0, -30.0,
             -COURT_HALF, -COURT_WIDTH / 2, 0.0],
            dtype=np.float32,
        )
    if high is None:
        high = np.array(
            [COURT_HALF, COURT_WIDTH / 2, 10.0,
             30.0, 30.0, 30.0,
             COURT_HALF, COURT_WIDTH / 2, 3.0],
            dtype=np.float32,
        )
    return 2.0 * (state_vec - low) / (high - low + 1e-8) - 1.0


def generate_random_state(rng=None):
    """
    随机生成一个合法的游戏状态（用于训练数据生成）。

    参数:
        rng: numpy 随机数生成器（可选）
    返回:
        GameState
    """
    if rng is None:
        rng = np.random.default_rng()

    ball_pos = np.array([
        rng.uniform(-COURT_HALF, COURT_HALF),
        rng.uniform(-COURT_WIDTH / 2, COURT_WIDTH / 2),
        rng.uniform(0.5, 5.0),
    ], dtype=np.float32)

    ball_vel = np.array([
        rng.uniform(-10.0, 10.0),
        rng.uniform(-10.0, 10.0),
        rng.uniform(-5.0, 5.0),
    ], dtype=np.float32)

    robot_pos = np.array([
        rng.uniform(-COURT_HALF, 0.0),   # 机器人在己方半场
        rng.uniform(-COURT_WIDTH / 2, COURT_WIDTH / 2),
        rng.uniform(0.0, 2.0),
    ], dtype=np.float32)

    return GameState(ball_pos, ball_vel, robot_pos)


def compute_optimal_action(state):
    """
    基于物理规则计算最优击球动作（用于生成监督学习标签）。

    策略：将球击向对方场地中央，选择合适的角度和力度。

    参数:
        state (GameState): 当前游戏状态
    返回:
        np.ndarray: 动作向量 [horizontal_angle, vertical_angle, force]
    """
    # 目标落点：对方场地中央
    target_x = COURT_HALF / 2.0
    target_y = 0.0

    dx = target_x - state.robot_pos[0]
    dy = target_y - state.robot_pos[1]
    horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)

    # 水平角度：朝向目标
    horizontal_angle = np.arctan2(dx, dy)
    horizontal_angle = float(np.clip(horizontal_angle, ACTION_LOW[0], ACTION_HIGH[0]))

    # 垂直角度：确保过网，取固定仰角
    vertical_angle = float(np.clip(np.deg2rad(30.0), ACTION_LOW[1], ACTION_HIGH[1]))

    # 力度：依据水平距离估算，使球能到达目标
    # 使用 max(..., 0.1) 防止 sin 值过小时产生极大的力度估算
    raw_force = np.sqrt(horizontal_dist * GRAVITY / max(np.sin(2 * vertical_angle), 0.1))
    force = float(np.clip(raw_force, ACTION_LOW[2], ACTION_HIGH[2]))

    return np.array([horizontal_angle, vertical_angle, force], dtype=np.float32)
