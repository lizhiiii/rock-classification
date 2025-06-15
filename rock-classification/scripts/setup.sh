#!/bin/bash

# 岩石分类项目环境设置脚本
# 使用方法: bash scripts/setup.sh

echo "===================================="
echo "岩石分类项目环境设置"
echo "===================================="

# 检查操作系统
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac
echo "检测到操作系统: ${MACHINE}"

# 检查Python版本
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "错误: 未找到Python安装"
    echo "请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(${PYTHON_CMD} --version 2>&1 | awk '{print $2}')
echo "Python版本: ${PYTHON_VERSION}"

# 检查Python版本是否满足要求
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "错误: Python版本过低，需要Python 3.8+"
    exit 1
fi

# 选择环境管理器
echo ""
echo "选择环境管理方式:"
echo "1) Conda (推荐)"
echo "2) Python虚拟环境 (venv)"
echo "3) 跳过环境设置，直接安装依赖"
read -p "请选择 (1-3): " ENV_CHOICE

case $ENV_CHOICE in
    1)
        # Conda环境
        if ! command -v conda &> /dev/null; then
            echo "错误: 未安装Conda"
            echo "请先安装Miniconda或Anaconda"
            exit 1
        fi
        
        echo "创建Conda环境..."
        conda create -n rock-cls python=3.10 -y
        echo "激活环境: conda activate rock-cls"
        echo "然后重新运行此脚本选择选项3"
        exit 0
        ;;
    2)
        # Python虚拟环境
        echo "创建Python虚拟环境..."
        ${PYTHON_CMD} -m venv venv
        
        if [ "${MACHINE}" = "Linux" ] || [ "${MACHINE}" = "Mac" ]; then
            echo "激活环境: source venv/bin/activate"
            source venv/bin/activate
        else
            echo "激活环境: venv\\Scripts\\activate"
        fi
        ;;
    3)
        echo "跳过环境设置..."
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

# 安装依赖
echo ""
echo "安装Python依赖包..."

# 升级pip
${PYTHON_CMD} -m pip install --upgrade pip

# 安装PyTorch (根据系统自动选择)
echo "安装PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，安装CUDA版本PyTorch"
    ${PYTHON_CMD} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "未检测到NVIDIA GPU，安装CPU版本PyTorch"
    ${PYTHON_CMD} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo "安装其他依赖包..."
${PYTHON_CMD} -m pip install -r requirements.txt

# 创建必要目录
echo ""
echo "创建项目目录结构..."
mkdir -p data/train_val/train
mkdir -p data/train_val/val
mkdir -p data/test/test_images
mkdir -p checkpoints
mkdir -p logs/tensorboard
mkdir -p outputs
mkdir -p assets

# 验证安装
echo ""
echo "验证安装..."
${PYTHON_CMD} -c "
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
print('✅ 核心依赖包安装成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'TIMM版本: {timm.__version__}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
else:
    print('使用CPU模式')
"

# 下载预训练模型（可选）
echo ""
read -p "是否下载ConvNeXt V2 Large预训练模型? (y/n): " DOWNLOAD_MODEL

if [ "$DOWNLOAD_MODEL" = "y" ] || [ "$DOWNLOAD_MODEL" = "Y" ]; then
    echo "下载预训练模型..."
    echo "注意: 这个模型约1.7GB，下载可能需要一些时间"
    
    # 这里需要实际的下载链接
    echo "请手动下载ConvNeXt V2 Large模型文件到checkpoints目录"
    echo "模型名称: convnextv2_large.fcmae_ft_in22k_in1k_384.bin"
    echo "下载链接: https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
fi

echo ""
echo "===================================="
echo "环境设置完成！"
echo ""
echo "接下来的步骤:"
echo "1. 准备数据集到data目录"
echo "2. 下载预训练模型到checkpoints目录"
echo "3. 运行训练: bash scripts/train.sh"
echo "4. 运行推理: bash scripts/inference.sh"
echo ""
echo "详细说明请查看README.md"
echo "====================================" 