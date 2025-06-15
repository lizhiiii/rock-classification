#!/bin/bash

# 岩石分类项目推理脚本
# 使用方法: bash scripts/inference.sh

echo "===================================="
echo "岩石分类项目推理开始"
echo "===================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装或不在PATH中"
    exit 1
fi

# 检查必要文件
echo "检查必要文件..."

# 检查测试数据
if [ ! -f "data/test/test_ids.csv" ]; then
    echo "错误: 测试ID文件不存在: data/test/test_ids.csv"
    exit 1
fi

if [ ! -d "data/test/test_images" ]; then
    echo "错误: 测试图像目录不存在: data/test/test_images"
    exit 1
fi

# 检查预训练模型
if [ ! -f "checkpoints/convnextv2_large.fcmae_ft_in22k_in1k_384.bin" ]; then
    echo "错误: 预训练模型文件不存在"
    echo "请先下载ConvNeXtV2-Large预训练权重"
    exit 1
fi

# 检查训练好的模型
BEST_MODEL=$(ls checkpoint/best_parent_*.pth 2>/dev/null | tail -1)
if [ -z "$BEST_MODEL" ]; then
    echo "错误: 未找到训练好的模型文件"
    echo "请先运行训练脚本或确保checkpoint目录中有best_parent_*.pth文件"
    exit 1
fi

echo "使用模型: $BEST_MODEL"

# 创建输出目录
mkdir -p outputs

# 运行推理
echo "开始推理..."
python -c "
import sys
sys.path.append('.')

# 这里可以添加专门的推理代码
# 目前使用Jupyter notebook的推理部分
print('请在Jupyter notebook中运行推理部分的代码')
print('或者运行: jupyter notebook main.ipynb')
print('然后执行推理相关的单元格')
"

# 检查输出文件
if [ -f "outputs/submission.csv" ]; then
    echo "推理完成！"
    echo "输出文件: outputs/submission.csv"
    
    # 显示文件信息
    echo "文件大小: $(wc -l < outputs/submission.csv) 行"
    echo "前几行内容:"
    head -5 outputs/submission.csv
else
    echo "警告: 未生成submission.csv文件"
    echo "请检查推理过程中是否有错误"
fi

echo "===================================="
echo "推理脚本执行完成"
echo "====================================" 