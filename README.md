# Strike Strategy Learning Project

从车位姿 + 球轨迹，学习最优的击球方向和力度

## Project Structure

```
strike_learning/
├── src/                    # ROS 2 C++ 源代码
│   └── strike_data_collector_node.cpp
├── scripts/               # Python 脚本
│   ├── analyze_data.py
│   ├── train_policy.py
│   └── utils.py
├── data/                  # 数据存储
│   ├── strike_data_week1.csv      (采集的数据)
│   └── strike_data_final.csv      (合并的数据)
├── models/               # 训练好的模型
│   ├── strike_policy.pth           (PyTorch 模型)
│   └── strike_policy.pt            (TorchScript 格式)
└── docs/                # 文档
    ├── learning_path.md
    ├── hands_on_plan.md
    └── action_checklist.md
```

## Timeline

- **Week 1 (D1-7)**: 数据采集 - ROS 编程 + 手柄录制
- **Week 2 (D8-14)**: 神经网络基础 - PyTorch 训练
- **Week 3-4 (D15-28)**: 迭代优化 - 模型调优 + 部署

## Quick Start

### Week 1: Data Collection

1. **创建 ROS 2 节点**
   ```bash
   cd src
   # 编辑 strike_data_collector_node.cpp
   ```

2. **编译**
   ```bash
   cd /home/toe/strike_learning
   # 通过 colcon 或 cmake 编译
   ```

3. **运行采集**
   ```bash
   ros2 run [package] strike_data_collector_node
   ```

### Week 2: Training

1. **加载数据并训练**
   ```bash
   cd scripts
   python train_policy.py --data ../data/strike_data_week1.csv
   ```

2. **检查结果**
   查看输出的 `strike_policy.pth` 和损失曲线图

## Key Files

- `src/strike_data_collector_node.cpp` - 数据采集节点
- `scripts/train_policy.py` - PyTorch 训练脚本
- `scripts/analyze_data.py` - 数据分析脚本
- `models/strike_policy.pth` - 最终模型

## Success Milestones

- ✅ D5: 收集 100+ 条数据
- ✅ D10: 损失下降 (0.5 → 0.1)
- ✅ D14: MAE < 0.15
- ✅ D28: 模型可部署

---

更多细节见 `docs/` 文件夹
