# PoseGeneration-Core

# 元宇宙数字人人体姿态生成与驱动系统

> 毕业设计项目 | 人工智能 × 计算机图形学 × 元宇宙

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1.1](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![Unity 2022 LTS](https://img.shields.io/badge/Unity-2022%20LTS-black.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**一个基于深度学习，能够从高层语义指令（类别/文本）自动生成并驱动3D数字人动作的端到端系统。**

**系统演示GIF/截图位置** *(训练完成后，请在此处添加一个GIF或截图，这是最重要的视觉展示)*

## 📖 项目概述
在当前元宇宙及数字人产业中，高质量动作制作成本高昂、效率低下。本项目旨在研发一套 **“指令输入-算法生成-虚拟人驱动”** 的自动化系统，为元宇宙内容创作提供高效、低成本的解决方案。

### 🎯 核心特性
- **智能生成**：基于改进的条件生成对抗网络（cGAN），根据动作类别或简短文本描述，自动生成合理、多样的3D人体姿态序列。
- **物理合理**：融合时空平滑损失与生物力学约束，有效减少动作抖动与肢体扭曲。
- **完整闭环**：实现从PyTorch算法模型到Unity三维渲染引擎的完整技术演示闭环。
- **应用友好**：生成标准BVH动画文件，兼容Maya、Blender等主流三维软件。

## 🏗️ 系统架构
系统采用**前后端分离**的松耦合架构，通过标准化的BVH文件进行数据交换。

```mermaid
graph TD
    A[用户指令] --> B[Unity前端]
    B -- HTTP请求/BVH文件 --> C[Python后端]
    B --> D[渲染驱动]
    C --> E[深度学习推理]
    D --> F[3D数字人动画]
    E --> G[动作生成模型(cGAN)]
```


### 📂 核心模块
1.  **`/algorithm` - 深度学习模型**
    - **数据预处理**：清洗AMASS等数据集，标准化为22关节骨骼。
    - **生成模型**：以TCN为主干的cGAN，集成多目标损失函数(WGAN-GP, 平滑损失, 生物力学约束)。
    - **评估体系**：FID、多样性、条件匹配准确率等量化指标。
2.  **`/unity_demo` - Unity演示系统**
    - **场景与角色**：包含已绑定Humanoid骨骼的数字人模型与交互场景。
    - **驱动控制器**：核心脚本`AvatarController.cs`，负责加载BVH动画并驱动角色。
    - **用户界面**：UGUI面板，提供动作选择、播放控制等功能。
3.  **`/docs` - 项目文档** (包含本项目完整的SDS设计文档)

## 🚀 快速开始
### 环境准备
- **算法端**：
  - **Python 3.9**
  - **PyTorch 2.1.1** + **CUDA 12.1** 或 **11.8**（如需GPU加速）。为确保与PyTorch 2.1.1的最佳兼容性，建议优先从PyTorch官方安装命令生成页面获取对应您系统的精确命令。
  - 其他依赖：详见 `algorithm/requirements.txt`
- **渲染端**：Unity 2022.3 LTS 或更高版本
- **三维工具**：Blender 3.0+（用于检查BVH文件）

### 安装与运行（算法端）
```bash
# 1. 克隆仓库
git clone https://github.com/your-organization/PoseGeneration-Core.git
cd PoseGeneration-Core/algorithm

# 2. 【强烈推荐】创建Python虚拟环境以隔离依赖
# 使用 conda
conda create -n posegen python=3.9
conda activate posegen
# 或使用 venv
python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows: .\venv\Scripts\activate

# 3. 安装PyTorch
# 安装支持CUDA 11.8的版本（兼容性更广，更稳定）
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# 如果仅使用CPU（不推荐用于训练）
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# 4. 安装项目其余依赖
pip install -r requirements.txt

# 5. 下载并预处理数据（示例）
python data/preprocessing.py --dataset amass --output_dir ./processed

# 6. 进行模型推理（示例）
python inference/generate.py --label walking --output ./animations/walk.bvh


