# 快速开始指南

这是一个简化的快速开始指南，帮助您快速运行岩石分类项目。

## 🚀 5分钟快速运行

### 1. 克隆项目
```bash
git clone https://github.com/your-username/rock-classification.git
cd rock-classification
```

### 2. 一键环境设置
```bash
bash scripts/setup.sh
```
按照提示选择环境管理方式（推荐选择Conda）

### 3. 激活环境
```bash
# 如果使用Conda
conda activate rock-cls

# 如果使用虚拟环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 4. 准备数据
将您的数据按照以下结构放在`data`目录中：
```
data/
├── train_val/
│   ├── train/           # 训练图像
│   ├── val/             # 验证图像
│   ├── train_labels.csv # 训练标签
│   └── val_labels.csv   # 验证标签
└── test/
    ├── test_images/     # 测试图像
    └── test_ids.csv     # 测试ID
```

### 5. 下载预训练模型
下载ConvNeXtV2-Large预训练模型到`checkpoints/`目录：
```bash
mkdir -p checkpoints
# 下载模型文件（约1.7GB）
wget -O checkpoints/convnextv2_large.fcmae_ft_in22k_in1k_384.bin \
  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
```

### 6. 开始训练
```bash
jupyter notebook main.ipynb
```
在浏览器中打开，按顺序执行所有单元格即可开始训练。

## 🔧 自定义配置

### 修改训练参数
编辑`config.py`文件中的参数：
```python
BATCH_SIZE = 32    # 如果GPU内存不足，减小批次大小
NUM_EPOCHS = 30    # 减少训练轮数进行快速测试
BASE_LR = 1e-4     # 调整学习率
```

### 快速测试运行
如果只想快速测试代码：
```python
# 在config.py中设置
DEBUG = True
FAST_DEV_RUN = True
NUM_EPOCHS = 2
```

## 📊 监控训练

### 查看训练日志
```bash
tail -f logs/train_log.csv
```

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练进度
训练过程会实时显示：
- 当前epoch进度
- 训练和验证损失
- 验证准确率
- 混淆矩阵

## 🎯 常见问题解决

### CUDA内存不足
```python
# 减小批次大小
BATCH_SIZE = 16  # 或更小

# 使用梯度累积
# 在训练循环中添加梯度累积逻辑
```

### 训练速度慢
```python
# 减少worker数量
NUM_WORKERS = 4

# 使用更小的图像尺寸
IMG_SIZE = 192
```

### 模型不收敛
```python
# 降低学习率
BASE_LR = 1e-5

# 增加预热期
WARM_EPOCHS = 10
```

## 🔄 完整工作流程

1. **数据准备** → `data/` 目录
2. **环境设置** → `bash scripts/setup.sh`
3. **模型下载** → `checkpoints/` 目录
4. **开始训练** → `jupyter notebook main.ipynb`
5. **监控训练** → 查看日志和指标
6. **模型推理** → 运行推理单元格
7. **提交结果** → `outputs/submission.csv`

## 📝 输出文件

训练完成后会生成：
- `checkpoint/best_parent_*.pth` - 最佳父类分类模型
- `checkpoint/best_child_*.pth` - 最佳子类分类模型
- `logs/train_log.csv` - 训练历史记录
- `outputs/submission.csv` - 测试集预测结果

## 🆘 获取帮助

如果遇到问题：
1. 查看详细的`README.md`
2. 检查`logs/`目录中的错误日志
3. 在GitHub仓库提交Issue
4. 联系项目维护者

---

**祝您训练顺利！** 🎉 