# 岩石分类项目 (Rock Classification)

基于深度学习的岩石图像分层分类系统，使用ConvNeXtV2-Large主干网络实现高精度三分类识别。

## 🏆 项目概述

本项目实现了一个先进的岩石图像分类系统，采用层次化分类策略，同时预测父类（3类）和子类（详细类别）。通过使用最新的ConvNeXtV2-Large预训练模型作为特征提取器，结合多种深度学习技术，实现了**81.47%**的验证准确率。

### 主要特点
- **先进架构**: ConvNeXtV2-Large主干网络 + 双头分类器
- **层次分类**: 同时预测粗粒度父类和细粒度子类
- **高级技术**: 混合精度训练、指数移动平均(EMA)、标签平滑
- **数据增强**: RandAugment + MixUp/CutMix
- **测试时增强**: 多尺度 + 水平翻转TTA

## 📊 性能表现

| 指标 | 数值 |
|------|------|
| 最佳验证准确率 | **81.47%** |
| 训练数据集 | 102,213张图像 |
| 验证数据集 | 15,000张图像 |
| 子类总数 | 19,160类 |
| 训练设备 | NVIDIA L20 |

## 📁 项目结构

```
rock-classification/
├── README.md                 # 📋 项目说明文档
├── main.ipynb               # 🔑 主训练和推理代码
├── requirements.txt          # 📦 Python依赖包
├── config.py                # ⚙️ 配置文件
├── train_log.csv            # 📈 训练历史记录
├── submission.csv           # 📄 生成的提交文件
├── 详细要求.md              # 📝 项目要求说明
├── 项目介绍.pdf             # 📖 项目介绍文档
├── QUICKSTART.md            # 🚀 快速开始指南
├── LICENSE                  # 📜 许可证
├── .gitignore              # 🚫 Git忽略文件
├── checkpoint/             # 📁 模型检查点目录（空）
├── logs/                   # 📁 日志目录
├── data/                   # 📁 数据目录（空）
├── src/                    # 📁 源代码目录
│   └── __init__.py
└── scripts/                # 📁 脚本目录
    ├── setup.sh            # 环境设置脚本
    ├── train.sh            # 训练脚本
    └── inference.sh        # 推理脚本
```

## 🚀 快速开始

### 🎓 南科大统计与数据科学系用户

**如果您是南科大统计与数据科学系相关人员：**

1. **直接上传文件到服务器**
   ```bash
   # 将main.ipynb上传到: /shareddata/project/dataset/
   cd /shareddata/project/dataset
   ```

2. **安装依赖库**
   ```bash
   pip install -r requirements.txt
   ```

3. **下载预训练模型**
   ```bash
   # 下载ConvNeXtV2-Large预训练模型到当前目录
   wget -O convnextv2_large.fcmae_ft_in22k_in1k_384.bin \
     "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
   ```

4. **开始训练**
   ```bash
   jupyter notebook main.ipynb
   # 按顺序执行所有单元格即可开始训练
   ```

---

### 🌐 其他用户使用指南

**如果您不是南科大统计与数据科学系人员：**

### 1. 环境配置

```bash
git clone https://github.com/your-username/rock-classification.git
cd rock-classification
```

**使用Conda (推荐)**
```bash
conda create -n rock-cls python=3.10
conda activate rock-cls
pip install -r requirements.txt
```

**使用虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. ⚠️ 重要：路径修改

**由于main.ipynb中硬编码了服务器路径，非南科大用户必须修改路径！**

打开 `main.ipynb`，找到第2个代码单元格，将：
```python
# 原始路径（服务器专用）
DATA_ROOT = "/shareddata/project/dataset"
```

修改为：
```python
# 修改为本地路径
DATA_ROOT = "data"  # 或您的数据存放路径
```

### 3. 数据集下载

**通过网盘下载完整数据集：**

🔗 **数据集下载地址**: https://pan.quark.cn/s/6dc546e5aae4#/list/share

下载后，按以下结构组织数据：
```
data/
├── train_val/
│   ├── train/              # 训练图像 (102,213张)
│   ├── val/                # 验证图像 (15,000张)
│   ├── train_labels.csv    # 训练标签
│   └── val_labels.csv      # 验证标签
└── test/
    ├── test_images/        # 测试图像
    └── test_ids.csv        # 测试ID
```

### 4. 模型文件获取

#### 4.1 预训练模型下载
```bash
# 从官网下载ConvNeXtV2-Large预训练模型
wget -O convnextv2_large.fcmae_ft_in22k_in1k_384.bin \
  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
```

#### 4.2 最佳训练模型下载（可选）
如果您想直接使用已训练好的最佳模型：

🔗 **最佳模型下载**: https://pan.baidu.com/s/1w3FRDxs_puIBaLQitGWxzA?pwd=vbxu  
**提取码**: `vbxu`

下载 `best_parent_0.8185.pth` 并放入 `checkpoint/` 目录。

### 5. 开始训练
```bash
jupyter notebook main.ipynb
# 按顺序执行所有单元格
```

## 📈 训练结果分析

### 性能指标
根据 `train_log.csv` 记录，训练过程中的关键指标：

- **最佳父类准确率**: 81.47% (Epoch 15)
- **训练时间**: 约19分钟/epoch (NVIDIA L20)
- **收敛情况**: 在前15个epoch内达到最佳性能
- **数据增强效果**: MixUp/CutMix在前80%训练期间有效

### 训练曲线特点
- 学习率调度：线性预热 + 余弦退火
- 损失下降稳定，从0.8023降至0.6401
- 验证准确率稳步提升，从69.30%升至81.47%
- EMA技术有效提升模型泛化能力

## ⚙️ 技术细节

### 模型架构
```python
# 主干网络
backbone = ConvNeXtV2-Large (预训练于ImageNet-22K)
# 分类头
head_parent = Linear(feat_dim, 3)      # 父类分类器
head_child = Linear(feat_dim, 19160)   # 子类分类器
```

### 核心技术
- **混合精度训练**: AMP加速训练，节省显存
- **指数移动平均**: EMA权重平滑，提升泛化性能
- **测试时增强**: 多尺度(224,256,288) + 水平翻转
- **数据增强**: RandAugment + MixUp/CutMix组合
- **标签平滑**: 0.1平滑系数，减少过拟合

### 训练配置
| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 64 | 批次大小 |
| Learning Rate | 4e-5 | 主干网络学习率 |
| Weight Decay | 1e-4 | 权重衰减 |
| Warm Epochs | 5 | 预热轮数 |
| EMA Decay | 0.9999 | 指数移动平均衰减 |

## 🔧 故障排除

### 常见问题

**1. CUDA内存不足**
```python
# 修改main.ipynb中的批次大小
BATCH_SIZE = 32  # 或更小
```

**2. 路径错误**
```python
# 确保修改了DATA_ROOT路径
DATA_ROOT = "data"  # 本地路径
```

**3. 模型文件缺失**
```bash
# 确保下载了预训练模型
ls -la convnextv2_large.fcmae_ft_in22k_in1k_384.bin
```

**4. 依赖库问题**
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

## 📊 文件说明

### 核心文件
- `main.ipynb`: 完整的训练和推理代码
- `train_log.csv`: 详细的训练历史记录
- `requirements.txt`: 项目依赖包列表
- `config.py`: 配置参数（实际以main.ipynb为准）

### 生成文件
- `submission.csv`: 测试集预测结果
- `checkpoint/best_parent_*.pth`: 最佳模型权重
- `train_log.csv`: 训练过程日志

## 🔗 资源链接

### 数据和模型获取
- **数据集**: https://pan.quark.cn/s/6dc546e5aae4#/list/share
- **最佳模型**: https://pan.baidu.com/s/1w3FRDxs_puIBaLQitGWxzA?pwd=vbxu (提取码: vbxu)
- **预训练模型**: https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt

### 参考资料
- ConvNeXtV2 论文: https://arxiv.org/abs/2301.00808
- timm 模型库: https://github.com/rwightman/pytorch-image-models
- PyTorch EMA: https://github.com/fadel/pytorch_ema

## 📝 使用步骤总结

1. **克隆项目** → `git clone ...`
2. **安装环境** → `pip install -r requirements.txt`
3. **修改路径** → 编辑main.ipynb中的DATA_ROOT
4. **下载数据** → 从网盘获取数据集
5. **下载模型** → 获取预训练模型文件
6. **开始训练** → `jupyter notebook main.ipynb`
7. **生成结果** → 运行推理单元格得到submission.csv

## 🤝 贡献与联系

如有任何问题或建议，请：
- 提交 GitHub Issue
- 发送邮件联系项目维护者

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

⭐ **如果这个项目对您有帮助，请给个Star！** ⭐ 