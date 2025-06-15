#!/bin/bash

# 岩石分类项目训练脚本
# 使用方法: bash scripts/train.sh

echo "===================================="
echo "岩石分类项目训练开始"
echo "===================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装或不在PATH中"
    exit 1
fi

# 检查必要的包
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import timm; print(f'TIMM版本: {timm.__version__}')"

# 检查GPU
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('警告: 未检测到GPU，将使用CPU训练（速度会很慢）')
"

# 创建必要目录
echo "创建必要目录..."
mkdir -p data/train_val/train
mkdir -p data/train_val/val
mkdir -p data/test/test_images
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs

# 检查数据是否存在
if [ ! -f "data/train_val/train_labels.csv" ]; then
    echo "警告: 训练标签文件不存在: data/train_val/train_labels.csv"
fi

if [ ! -f "checkpoints/convnextv2_large.fcmae_ft_in22k_in1k_384.bin" ]; then
    echo "警告: 预训练模型文件不存在"
    echo "请下载ConvNeXtV2-Large预训练权重到checkpoints目录"
fi

# 开始训练
echo "开始训练..."
python -c "
import sys
sys.path.append('.')
from config import setup_directories, validate_config
setup_directories()
validate_config()
"

# 使用Jupyter执行训练
echo "启动Jupyter Notebook进行训练..."
echo "请在浏览器中打开并运行 main.ipynb"
jupyter notebook main.ipynb

echo "===================================="
echo "训练完成！"
echo "请查看以下文件："
echo "- 训练日志: logs/train_log.csv"
echo "- 最佳模型: checkpoint/best_parent_*.pth"
echo "- 提交文件: outputs/submission.csv"
echo "====================================" 