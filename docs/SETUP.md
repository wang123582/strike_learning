# 软件和依赖安装指南

## 环境要求

- **操作系统**: Ubuntu 20.04 / 22.04（兼容 ROS 2）
- **Python**: 3.8+
- **ROS 2**: Foxy / Humble（应该已有）

---

## 📦 需要安装的依赖

### 1️⃣ Python 科学计算栈（用于训练和数据分析）

```bash
pip install --upgrade pip

# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn

# 可选但推荐
pip install jupyter jupyterlab  # 用于交互式开发
pip install tensorboard         # 用于可视化训练
```

**验证安装**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pandas; print(f'Pandas {pandas.__version__}')"
```

---

### 2️⃣ ROS 2 开发工具（应该已有，但检查一下）

```bash
# 如果还没装，安装 ROS 2 构建工具
sudo apt install -y \
  build-essential \
  cmake \
  git \
  ros-humble-ament-cmake \
  ros-humble-std-msgs \
  ros-humble-geometry-msgs \
  ros-humble-nav-msgs
```

**验证安装**:
```bash
ros2 --version
ament --help
```

---

### 3️⃣ C++ 编译器和工具

```bash
# 如果还没装
sudo apt install -y g++ cmake git
```

---

## 🚀 快速安装脚本

将以下内容保存为 `install_dependencies.sh`：

```bash
#!/bin/bash
set -e

echo "🔧 Installing Strike Learning Dependencies..."

# Python 依赖
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn

# ROS 2 依赖（可选，如果没装过）
echo "📦 Installing ROS 2 packages..."
sudo apt update
sudo apt install -y \
  ros-humble-std-msgs \
  ros-humble-geometry-msgs \
  ros-humble-nav-msgs

echo "✅ Installation complete!"
echo ""
echo "验证 PyTorch:"
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
echo ""
echo "验证 Pandas:"
python -c "import pandas; print(f'✅ Pandas {pandas.__version__}')"
```

运行:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

---

## 🐍 创建虚拟环境（推荐但非必须）

如果想隔离项目依赖：

```bash
cd /home/toe/strike_learning

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install torch numpy pandas matplotlib scikit-learn
```

每次使用前记得激活虚拟环境:
```bash
source /home/toe/strike_learning/venv/bin/activate
```

---

## 📋 依赖清单

| 软件/库 | 版本 | 用途 | 安装命令 |
|--------|------|------|---------|
| Python | 3.8+ | 编程语言 | 系统自带 |
| PyTorch | 2.0+ | 神经网络框架 | `pip install torch` |
| NumPy | 1.21+ | 数值计算 | `pip install numpy` |
| Pandas | 1.3+ | 数据处理 | `pip install pandas` |
| Matplotlib | 3.5+ | 数据可视化 | `pip install matplotlib` |
| Scikit-Learn | 1.0+ | 机器学习工具 | `pip install scikit-learn` |
| ROS 2 | Humble | 机器人框架 | 应该已有 |
| C++ 编译器 | g++ 9+ | 编译 CPP 代码 | 系统通常自带 |

---

## ✅ 验证安装

运行以下命令检查所有依赖是否正确安装：

```bash
#!/bin/bash
echo "=== 验证安装 ==="

# PyTorch
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch 未安装"

# NumPy
python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__}')" || echo "❌ NumPy 未安装"

# Pandas
python3 -c "import pandas; print(f'✅ Pandas {pandas.__version__}')" || echo "❌ Pandas 未安装"

# Matplotlib
python3 -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__}')" || echo "❌ Matplotlib 未安装"

# Scikit-learn
python3 -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__}')" || echo "❌ Scikit-learn 未安装"

# ROS 2
ros2 --version && echo "✅ ROS 2 已安装" || echo "❌ ROS 2 未安装"

echo "=== 验证完成 ==="
```

---

## 🆘 常见问题

### Q: PyTorch 安装很慢？
**A**: 换国内镜像源：
```bash
pip config set global.index-url https://pypi.tsinghua.edu.cn/simple
```

### Q: 无法安装 PyTorch（特别是在 ARM 架构）?
**A**: 尝试用 CPU 版本并跳过 CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: ROS 2 相关的头文件找不到？
**A**: 确保装了开发文件：
```bash
sudo apt install -y ros-humble-std-msgs-dev ros-humble-geometry-msgs-dev
```

---

## 📝 安装完成后

1. ✅ 验证所有依赖都已安装
2. ✅ 激活虚拟环境（如果使用）
3. ✅ 进入项目目录：`cd /home/toe/strike_learning`
4. ✅ 准备开始 Week 1 数据采集

---

## 下一步

完成安装后，回到 `PROJECT.md` 并按照时间表开始 Week 1！
