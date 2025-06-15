# 快速开始指南

这是一个简化的快速开始指南，帮助您快速运行岩石分类项目。

## 🚀 5分钟快速运行

### 1. 克隆项目
```bash
git clone https://github.com/your-username/rock-classification.git
cd rock-classification
```

### 2. 环境设置
```bash
# 使用Conda (推荐)
conda create -n rock-cls python=3.10
conda activate rock-cls
pip install -r requirements.txt

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. ⚠️ 重要：修改数据路径
**非南科大用户必须修改main.ipynb中的路径！**

打开 `main.ipynb`，在第2个代码单元格中将：
```python
DATA_ROOT = "/shareddata/project/dataset"  # 服务器路径
```
修改为：
```python
DATA_ROOT = "data"  # 本地路径
```

### 4. 准备数据
从网盘下载数据集并按以下结构组织：

🔗 **数据集下载**: https://pan.quark.cn/s/6dc546e5aae4#/list/share

```
data/
├── train_val/
│   ├── train/           # 训练图像 (102,213张)
│   ├── val/             # 验证图像 (15,000张)
│   ├── train_labels.csv # 训练标签
│   └── val_labels.csv   # 验证标签
└── test/
    ├── test_images/     # 测试图像
    └── test_ids.csv     # 测试ID
```

### 5. 下载预训练模型
```bash
# 下载ConvNeXtV2-Large预训练模型到项目根目录
wget -O convnextv2_large.fcmae_ft_in22k_in1k_384.bin \
  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
```

### 6. 开始训练
```bash
jupyter notebook main.ipynb
```
在浏览器中打开，按顺序执行所有单元格即可开始训练。

## 🔧 快速调优

### 减少GPU内存使用
如果遇到CUDA内存不足，修改main.ipynb中的批次大小：
```python
BATCH_SIZE = 32  # 或16，根据GPU显存调整
```

### 快速测试运行
想要快速测试代码，可以减少训练轮数：
```python
NUM_EPOCHS = 5  # 减少到5轮快速测试
```

## 📊 监控训练

### 查看训练进度
训练过程会实时显示：
- 当前epoch进度条
- 训练和验证损失
- 验证准确率（目标：>81%）
- 混淆矩阵

### 查看训练历史
```bash
# 查看训练日志
cat train_log.csv
# 或实时监控
tail -f train_log.csv
```

### GPU使用监控
```bash
watch -n 1 nvidia-smi
```

## 🎯 常见问题快速解决

### 1. 路径错误
```
FileNotFoundError: No such file or directory: '/shareddata/...'
```
**解决**: 确保已修改main.ipynb中的DATA_ROOT路径

### 2. 内存不足
```
CUDA out of memory
```
**解决**: 减小BATCH_SIZE到32或16

### 3. 模型文件缺失
```
FileNotFoundError: convnextv2_large.fcmae_ft_in22k_in1k_384.bin
```
**解决**: 确保已下载预训练模型到项目根目录

### 4. 数据集缺失
```
FileNotFoundError: data/train_val/train_labels.csv
```
**解决**: 从网盘下载数据集并正确组织文件结构

## 🚀 预训练模型选项

### 选项1: 从零开始训练
下载预训练模型，完整训练约60个epoch：
```bash
wget -O convnextv2_large.fcmae_ft_in22k_in1k_384.bin \
  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
```

### 选项2: 使用已训练模型
下载已训练好的最佳模型：

🔗 **最佳模型**: https://pan.baidu.com/s/1w3FRDxs_puIBaLQitGWxzA?pwd=vbxu (提取码: vbxu)

```bash
# 创建checkpoint目录并放入下载的模型
mkdir -p checkpoint
# 将best_parent_0.8185.pth放入checkpoint/目录
```

## 📈 期望结果

### 训练指标
- **最佳验证准确率**: 81.47%
- **训练时间**: ~19分钟/epoch (NVIDIA L20)
- **总训练时间**: 约5-6小时 (完整60轮)
- **收敛轮数**: 通常在15-20轮达到最佳性能

### 生成文件
训练完成后会在项目根目录生成：
- `train_log.csv` - 详细训练记录
- `submission.csv` - 测试集预测结果
- `checkpoint/best_parent_*.pth` - 最佳模型权重

## 🔄 完整工作流程总结

1. **克隆项目** → `git clone ...`
2. **安装环境** → `conda create -n rock-cls python=3.10`
3. **修改路径** → 编辑main.ipynb中的DATA_ROOT
4. **下载数据** → 从网盘获取数据集
5. **下载模型** → 获取预训练模型文件
6. **开始训练** → `jupyter notebook main.ipynb`
7. **等待收敛** → 监控训练进度
8. **获取结果** → submission.csv文件

## 🆘 获取帮助

如果遇到问题：
1. 查看详细的 [README.md](README.md)
2. 检查train_log.csv中的错误信息
3. 确认数据路径和文件结构正确
4. 在GitHub仓库提交Issue

---

**祝您训练顺利！达成81%+准确率目标！** 🎉 