# strike_learning
排球机器人的策略

## 通过神经网络实现击球决策

本项目使用全连接前馈神经网络（PyTorch）学习最优击球策略。

### 文件结构

| 文件 | 说明 |
|------|------|
| `environment.py` | 游戏状态（`GameState`）、击球动作（`StrikeAction`）及物理辅助函数 |
| `model.py` | 击球决策神经网络 `StrikeNet` |
| `train.py` | 生成训练数据并训练模型 |
| `decision.py` | 加载已训练模型，对外提供 `StrikeDecisionMaker` 推理接口 |
| `test_model.py` | 各模块单元测试 |

### 快速开始

```bash
# 安装依赖
pip install torch numpy pytest

# 训练模型（约 50 轮，保存为 strike_model.pth）
python train.py

# 使用训练好的模型进行推理示例
python decision.py

# 运行所有单元测试
pytest test_model.py -v
```

### 网络结构

- **输入**：9 维归一化状态向量 `[ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, robot_x, robot_y, robot_z]`
- **隐藏层**：2 × 128 神经元，ReLU 激活
- **输出**：3 维动作向量 `[horizontal_angle, vertical_angle, force]`，通过 Tanh + 线性映射约束到合法范围

### 训练方式

使用物理规则（将球击向对方场地中央）自动生成 5 万条监督学习样本，采用 Adam 优化器与 MSE 损失函数训练 50 轮，最终损失可降至 `~0.00016`。
