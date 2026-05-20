#!/usr/bin/env python3
"""
数据分析脚本
用于可视化和检查采集到的数据质量
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_analyze(csv_path):
    """加载 CSV 并分析"""
    print(f"Loading {csv_path}...")
    data = pd.read_csv(csv_path)
    
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # 基本统计
    print(f"\nBasic Statistics:")
    print(data.describe())
    
    # 检查缺失值
    print(f"\nMissing values:")
    print(data.isnull().sum())
    
    return data


def plot_distributions(data, save_path="data_distribution.png"):
    """绘制数据分布"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle("Data Distribution", fontsize=16)
    
    columns = ['x_robot', 'y_robot', 'theta_robot', 'x_ball', 'y_ball', 'vx_ball', 'vy_ball', 'strike_angle']
    
    for i, col in enumerate(columns):
        ax = axes[i // 4, i % 4]
        if col in data.columns:
            ax.hist(data[col], bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(col)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    
    # 最后一个子图用于 strike_force
    if 'strike_force' in data.columns:
        axes[1, 3].hist(data['strike_force'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 3].set_title('strike_force')
        axes[1, 3].set_xlabel("Value")
        axes[1, 3].set_ylabel("Frequency")
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Distribution plot saved to {save_path}")
    plt.close()


def plot_correlations(data, save_path="correlations.png"):
    """绘制特征相关性"""
    # 只选择数值列
    numeric_data = data.select_dtypes(include=[np.number])
    
    corr = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # 设置标签
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    # 添加数值
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im)
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Correlation plot saved to {save_path}")
    plt.close()


def plot_scatter(data, save_path="scatter_plots.png"):
    """绘制机器人位置和击球方向的散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 机器人位置
    if 'x_robot' in data.columns and 'y_robot' in data.columns:
        ax = axes[0]
        sc = ax.scatter(data['x_robot'], data['y_robot'], 
                       c=data.get('strike_angle', range(len(data))),
                       cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel("X Robot")
        ax.set_ylabel("Y Robot")
        ax.set_title("Robot Positions (colored by strike_angle)")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax)
    
    # 击球角度 vs 力度
    if 'strike_angle' in data.columns and 'strike_force' in data.columns:
        ax = axes[1]
        ax.scatter(data['strike_angle'], data['strike_force'], 
                  s=50, alpha=0.6, c=range(len(data)), cmap='plasma')
        ax.set_xlabel("Strike Angle")
        ax.set_ylabel("Strike Force")
        ax.set_title("Strike Parameters")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Scatter plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze strike data")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    
    args = parser.parse_args()
    
    # 加载和分析
    data = load_and_analyze(args.data)
    
    # 生成图表
    plot_distributions(data)
    plot_correlations(data)
    plot_scatter(data)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
