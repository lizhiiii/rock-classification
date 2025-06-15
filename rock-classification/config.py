"""
岩石分类项目配置文件
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
# 数据根目录
DATA_ROOT = "data"  # 修改为相对路径

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
# 预训练模型
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
CKPT_PATH = os.path.join("checkpoints", "convnextv2_large.fcmae_ft_in22k_in1k_384.bin")

# 类别数量
NUM_PARENT = 3  # 父类数量
# NUM_CHILD 将在运行时从数据中计算

# ==============================================================
# 4. 训练超参数
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
# 图像归一化参数（ImageNet预训练）
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
# 检查点保存路径
CHECKPOINT_DIR = "checkpoint"

# 日志保存路径
LOG_DIR = "logs"
TRAIN_LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")

# 输出路径
OUTPUT_DIR = "outputs"
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

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
# 10. 创建必要目录
# ==============================================================
def setup_directories():
    """创建必要的目录"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

# ==============================================================
# 11. 配置验证
# ==============================================================
def validate_config():
    """验证配置的有效性"""
    # 检查重要文件是否存在
    if not os.path.exists(CKPT_PATH):
        print(f"警告: 预训练模型文件不存在: {CKPT_PATH}")
    
    # 检查数据路径
    if not os.path.exists(DATA_ROOT):
        print(f"警告: 数据根目录不存在: {DATA_ROOT}")
    
    # 打印配置信息
    print("=" * 60)
    print("岩石分类项目配置")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"基础学习率: {BASE_LR}")
    print(f"预训练模型: {MODEL_NAME}")
    print(f"使用混合精度: {USE_AMP}")
    print(f"使用TTA: {USE_TTA}")
    print("=" * 60)

if __name__ == "__main__":
    setup_directories()
    validate_config() 