"""
岩石分类项目配置文件
注意：main.ipynb中有自己的参数设置，实际运行以main.ipynb为准！
本文件仅作为参考配置。
"""
import os
import torch

# ==============================================================
# 1. 基础配置
# ==============================================================
# 随机种子
SEED = 114514

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================
# 2. 数据路径配置
# ==============================================================
# 数据根目录 (本地用户使用，南科大用户请使用服务器路径)
DATA_ROOT = "data"  # 本地相对路径
# 南科大服务器路径: "/shareddata/project/dataset"

# 训练和验证数据
TRAIN_CSV = os.path.join(DATA_ROOT, "train_val", "train_labels.csv")
VAL_CSV = os.path.join(DATA_ROOT, "train_val", "val_labels.csv")
IMG_DIR_TRAIN = os.path.join(DATA_ROOT, "train_val", "train")
IMG_DIR_VAL = os.path.join(DATA_ROOT, "train_val", "val")

# 测试数据
TEST_IMGDIR = os.path.join(DATA_ROOT, "test", "test_images")
TEST_CSV = os.path.join(DATA_ROOT, "test", "test_ids.csv")

# ==============================================================
# 3. 模型配置
# ==============================================================
# 预训练模型 (main.ipynb中实际使用项目根目录)
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
CKPT_PATH = "convnextv2_large.fcmae_ft_in22k_in1k_384.bin"  # 项目根目录

# 类别数量
NUM_PARENT = 3  # 父类数量
# NUM_CHILD 将在运行时从数据中计算 (实际为19160)

# ==============================================================
# 4. 训练超参数 (与main.ipynb保持一致)
# ==============================================================
# 基础训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 60
BASE_LR = 2e-4
WARM_EPOCHS = 5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

# EMA相关
EMA_DECAY = 0.9999

# MixUp/CutMix
MIXUP_CUTMIX_STOP = int(NUM_EPOCHS * 0.8)  # 前80%epoch使用数据增强

# 早停
PATIENCE = 5

# ==============================================================
# 5. 数据增强配置
# ==============================================================
# 图像归一化参数（与main.ipynb保持一致）
MEAN = [0.46798, 0.45764, 0.44035]
STD = [0.18461, 0.18712, 0.19482]

# 图像尺寸
IMG_SIZE = 224

# RandAugment参数
RANDAUG_NUM_OPS = 2
RANDAUG_MAGNITUDE = 8

# ==============================================================
# 6. 推理配置
# ==============================================================
# TTA（测试时增强）
TTA_SCALES = [224, 256, 288]
USE_TTA = True

# ==============================================================
# 7. 输出路径配置
# ==============================================================
# 检查点保存路径 (main.ipynb中使用"checkpoint")
CHECKPOINT_DIR = "checkpoint"

# 日志保存路径 (main.ipynb中直接保存到根目录)
LOG_DIR = "logs"
TRAIN_LOG_FILE = "train_log.csv"  # 实际保存在项目根目录

# 输出路径 (main.ipynb中直接保存到根目录)
OUTPUT_DIR = "outputs"
SUBMISSION_FILE = "submission.csv"  # 实际保存在项目根目录

# ==============================================================
# 8. 系统配置
# ==============================================================
# 数据载入
NUM_WORKERS = 8
PIN_MEMORY = True

# 打印频率
PRINT_FREQ = 100

# 混合精度训练
USE_AMP = True

# ==============================================================
# 9. 调试配置
# ==============================================================
DEBUG = False
FAST_DEV_RUN = False  # 快速开发运行（仅训练几个batch）

# 验证频率
EVAL_EVERY = 1  # 每几个epoch验证一次

# ==============================================================
# 10. 重要提示
# ==============================================================
"""
⚠️ 重要提示：
1. 本配置文件仅供参考，实际运行请使用main.ipynb
2. main.ipynb中有完整的参数设置和训练逻辑
3. 非南科大用户请修改main.ipynb中的DATA_ROOT路径
4. 预训练模型文件应放在项目根目录
5. 最佳验证准确率：81.47%
"""

# ==============================================================
# 11. 创建必要目录
# ==============================================================
def setup_directories():
    """创建必要的目录"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# 12. 配置验证
# ==============================================================
def validate_config():
    """验证配置的有效性"""
    print("=" * 60)
    print("岩石分类项目配置 (参考)")
    print("=" * 60)
    print(f"⚠️  注意：实际运行请使用 main.ipynb")
    print(f"设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"基础学习率: {BASE_LR}")
    print(f"预训练模型: {MODEL_NAME}")
    print(f"使用混合精度: {USE_AMP}")
    print(f"使用TTA: {USE_TTA}")
    print(f"目标准确率: 81.47%")
    
    # 检查重要文件
    if os.path.exists(CKPT_PATH):
        print(f"✅ 预训练模型已存在: {CKPT_PATH}")
    else:
        print(f"❌ 预训练模型不存在: {CKPT_PATH}")
        print("   请下载: wget -O convnextv2_large.fcmae_ft_in22k_in1k_384.bin https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt")
    
    if os.path.exists(DATA_ROOT):
        print(f"✅ 数据目录已存在: {DATA_ROOT}")
    else:
        print(f"❌ 数据目录不存在: {DATA_ROOT}")
        print("   请从网盘下载数据集: https://pan.quark.cn/s/6dc546e5aae4#/list/share")
    
    print("=" * 60)

if __name__ == "__main__":
    setup_directories()
    validate_config() 