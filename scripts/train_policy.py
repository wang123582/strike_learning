#!/usr/bin/env python3
"""
最小化的 PyTorch 训练脚本
用于验证数据加载和模型训练流程

Usage:
    python train_policy.py --data ../data/strike_data_week1.csv --epochs 100 --lr 0.001
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch scikit-learn pandas matplotlib")
    sys.exit(1)


class StrikePolicy(nn.Module):
    """7 维输入 → 2 维输出 的小网络"""
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(csv_path):
    """加载 CSV 数据"""
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    # 选择特征和标签
    feature_cols = ['x_robot', 'y_robot', 'theta_robot', 'x_ball', 'y_ball', 'vx_ball', 'vy_ball']
    label_cols = ['strike_angle', 'strike_force']
    
    # 检查列是否存在
    for col in feature_cols + label_cols:
        if col not in data.columns:
            print(f"Warning: Column '{col}' not found. Using zeros.")
            data[col] = 0.0
    
    X = torch.FloatTensor(data[feature_cols].values)
    Y = torch.FloatTensor(data[label_cols].values)
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Data statistics:")
    print(f"  X mean={X.mean(dim=0)}, std={X.std(dim=0)}")
    print(f"  Y mean={Y.mean(dim=0)}, std={Y.std(dim=0)}")
    
    return X, Y


def train_model(X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001, batch_size=16):
    """训练模型"""
    print(f"\nTraining model for {epochs} epochs with lr={lr}...")
    
    # 初始化模型
    model = StrikePolicy()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 转为 DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 记录损失
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= X_train.size(0)
        train_losses.append(train_loss)
        
        # 测试阶段
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test)
            test_loss = criterion(pred_test, Y_test).item()
            test_losses.append(test_loss)
        
        # 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final test loss: {test_losses[-1]:.4f}")
    
    return model, train_losses, test_losses


def evaluate_model(model, X_test, Y_test):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mae = torch.abs(pred - Y_test).mean(dim=0)
        rmse = torch.sqrt(torch.pow(pred - Y_test, 2).mean(dim=0))
    
    print(f"\nModel Evaluation:")
    print(f"  MAE (angle, force): {mae[0]:.4f}, {mae[1]:.4f}")
    print(f"  RMSE (angle, force): {rmse[0]:.4f}, {rmse[1]:.4f}")
    
    return mae, rmse


def save_model(model, path="strike_policy.pth"):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to {path}")


def plot_loss(train_losses, test_losses, save_path="training_loss.png"):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(test_losses, label="Test Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train strike policy network")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--model_path", type=str, default="strike_policy.pth", help="Output model path")
    
    args = parser.parse_args()
    
    # 加载数据
    X, Y = load_data(args.data)
    
    # 分割训练/测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_split, random_state=42)
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 训练模型
    model, train_losses, test_losses = train_model(
        X_train, Y_train, X_test, Y_test,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    
    # 评估
    evaluate_model(model, X_test, Y_test)
    
    # 保存模型和结果
    save_model(model, args.model_path)
    plot_loss(train_losses, test_losses, "training_loss.png")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
