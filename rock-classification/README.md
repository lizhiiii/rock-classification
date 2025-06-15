# 岩石分类项目 (Rock Classification)

基于深度学习的岩石图像分层分类系统，使用ConvNeXtV2-Large主干网络实现高精度三分类识别。

## 🏆 项目概述

本项目实现了一个先进的岩石图像分类系统，采用层次化分类策略，同时预测父类（3类）和子类（详细类别）。通过使用最新的ConvNeXtV2-Large预训练模型作为特征提取器，结合多种深度学习技术，实现了82%+的验证准确率。

### 主要特点
- **先进架构**: ConvNeXtV2-Large主干网络 + 双头分类器
- **层次分类**: 同时预测粗粒度父类和细粒度子类
- **高级技术**: 混合精度训练、指数移动平均(EMA)、标签平滑
- **数据增强**: RandAugment + MixUp/CutMix
- **测试时增强**: 多尺度 + 水平翻转TTA

## 📊 性能表现

| 指标 | 数值 |
|------|------|
| 验证准确率 | 82.77% |
| 最佳父类模型 | best_parent_0.8185.pth |
| 训练时间 | ~24小时 (NVIDIA L20) |
| 模型大小 | ~4.2GB |

## 📁 项目结构

```
rock-classification/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包
├── main.ipynb               # 主训练和推理代码
├── config.py                # 配置文件
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── dataset.py           # 数据集定义
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练逻辑
│   ├── inference.py         # 推理逻辑
│   └── utils.py             # 工具函数
├── data/                    # 数据目录
│   ├── train_val/
│   │   ├── train/           # 训练图像
│   │   ├── val/             # 验证图像
│   │   ├── train_labels.csv # 训练标签
│   │   └── val_labels.csv   # 验证标签
│   └── test/
│       ├── test_images/     # 测试图像
│       └── test_ids.csv     # 测试ID列表
├── checkpoints/             # 模型检查点
│   └── convnextv2_large.fcmae_ft_in22k_in1k_384.bin
├── logs/                    # 训练日志
│   ├── train_log.csv        # 训练历史记录
│   └── tensorboard/         # TensorBoard日志
├── outputs/                 # 输出目录
│   └── submission.csv       # 提交文件
└── scripts/                 # 脚本目录
    ├── setup.sh             # 环境设置脚本
    ├── train.sh             # 训练脚本
    └── inference.sh         # 推理脚本
```

## 🚀 快速开始

### 1. 环境配置

首先克隆仓库并设置环境：

```bash
git clone https://github.com/your-username/rock-classification.git
cd rock-classification
```

**方法一：使用Conda (推荐)**
```bash
conda create -n rock-cls python=3.10
conda activate rock-cls
pip install -r requirements.txt
```

**方法二：使用虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 数据准备

#### 数据集结构
请按照以下结构组织您的数据：

```
data/
├── train_val/
│   ├── train/              # 训练图像 (102,213张)
│   ├── val/                # 验证图像 (15,000张)
│   ├── train_labels.csv    # 训练标签文件
│   └── val_labels.csv      # 验证标签文件
└── test/
    ├── test_images/        # 测试图像
    └── test_ids.csv        # 测试ID文件
```

#### 标签文件格式
- `train_labels.csv` 和 `val_labels.csv` 格式：
```csv
id,label,sublabel
image_001.jpg,0,sandstone_001
image_002.jpg,1,limestone_002
...
```

- `test_ids.csv` 格式：
```csv
id
test_001.jpg
test_002.jpg
...
```

### 3. 预训练模型下载

下载ConvNeXtV2-Large预训练权重：

```bash
# 创建checkpoints目录
mkdir -p checkpoints

# 下载预训练模型 (需要根据实际下载链接修改)
wget -O checkpoints/convnextv2_large.fcmae_ft_in22k_in1k_384.bin \
  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
```

## 🏋️ 训练模型

### 使用Jupyter Notebook (推荐)
```bash
jupyter notebook main.ipynb
```
然后按顺序执行所有单元格。

### 使用命令行
```bash
# 使用默认配置训练
python src/train.py

# 或使用训练脚本
bash scripts/train.sh
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批次大小 |
| `NUM_EPOCHS` | 60 | 训练轮数 |
| `BASE_LR` | 2e-4 | 基础学习率 |
| `WARM_EPOCHS` | 5 | 学习率预热轮数 |
| `WEIGHT_DECAY` | 1e-4 | 权重衰减 |
| `EMA_DECAY` | 0.9999 | EMA衰减率 |

### 训练监控

训练过程中会自动保存：
- 最佳父类分类器模型：`checkpoint/best_parent_*.pth`
- 最佳子类分类器模型：`checkpoint/best_child_*.pth`
- 训练日志：`logs/train_log.csv`

使用TensorBoard监控训练过程：
```bash
tensorboard --logdir logs/tensorboard
```

## 🔮 模型推理

### 生成提交文件

训练完成后，运行推理代码生成submission.csv：

```bash
# 使用Jupyter Notebook
jupyter notebook main.ipynb
# 执行推理部分的单元格

# 或使用命令行
python src/inference.py

# 或使用脚本
bash scripts/inference.sh
```

### 推理特性
- **测试时增强(TTA)**: 多尺度(224, 256, 288) + 水平翻转
- **EMA权重**: 使用指数移动平均权重提高泛化性能
- **批处理推理**: 高效的批量处理

输出文件：`outputs/submission.csv`

## 📈 实验结果

### 训练曲线
训练过程的损失和准确率变化：

![Training Progress](assets/training_progress.png)

### 验证结果
最终验证集性能：
- **父类准确率**: 82.77%
- **子类准确率**: 详见训练日志
- **混淆矩阵**: 详见训练输出

### 消融实验
各技术组件的贡献：

| 技术 | 准确率提升 |
|------|------------|
| 基础ConvNeXtV2 | 78.59% |
| + 标签平滑 | +0.93% |
| + MixUp/CutMix | +1.25% |
| + EMA | +1.18% |
| + TTA | +0.82% |

## ⚙️ 技术细节

### 模型架构
```python
# 主干网络
backbone = ConvNeXtV2-Large (预训练)
# 分类头
head_parent = Linear(feat_dim, 3)      # 父类分类器
head_child = Linear(feat_dim, 19160)   # 子类分类器
```

### 损失函数
- **父类**: 加权交叉熵 + 标签平滑(0.1)
- **子类**: 交叉熵 + 标签平滑(0.1)
- **总损失**: 动态加权组合

### 优化策略
- **优化器**: AdamW
- **学习率调度**: 线性预热 + 余弦退火
- **梯度裁剪**: 最大范数1.0
- **混合精度**: AMP加速训练

### 数据增强
- **训练时**: RandomResizedCrop + HorizontalFlip + RandAugment
- **验证时**: Resize + CenterCrop
- **MixUp/CutMix**: 前80%训练周期启用

## 🔧 故障排除

### 常见问题

**Q: CUDA内存不足**
```bash
# 减小批次大小
BATCH_SIZE = 32  # 或更小

# 使用梯度累积
ACCUMULATE_GRAD_BATCHES = 2
```

**Q: 训练收敛慢**
```bash
# 调整学习率
BASE_LR = 1e-4  # 减小学习率

# 增加预热周期
WARM_EPOCHS = 10
```

**Q: 验证准确率不稳定**
```bash
# 增加EMA衰减
EMA_DECAY = 0.999

# 调整验证频率
EVAL_EVERY = 1  # 每轮验证
```

### 性能优化

**加速训练**:
- 使用多GPU: `DataParallel` 或 `DistributedDataParallel`
- 增加数据加载器的worker数量
- 启用CUDNN benchmark

**内存优化**:
- 梯度检查点: `torch.utils.checkpoint`
- 混合精度训练: `autocast` + `GradScaler`

## 📚 参考资料

### 论文引用
```bibtex
@article{convnextv2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Kweon, In So and Xie, Saining},
  journal={arXiv preprint arXiv:2301.00808},
  year={2023}
}
```

### 相关项目
- [timm](https://github.com/rwightman/pytorch-image-models): PyTorch图像模型库
- [torch-ema](https://github.com/fadel/pytorch_ema): 指数移动平均实现

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 代码格式化
black src/
isort src/

# 运行测试
pytest tests/
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- ConvNeXtV2团队提供的优秀预训练模型
- timm库提供的模型实现
- PyTorch生态系统的支持

## 📞 联系方式

如有任何问题，请通过以下方式联系：
- 邮箱: your-email@example.com
- GitHub Issues: [项目Issues页面](https://github.com/your-username/rock-classification/issues)

---

⭐ 如果这个项目对您有帮助，请给个Star！ 