# ====== 训练参数 ======
MAX_EPOCHS = 50              # 最大训练轮数
BATCH_SIZE = 32              # 批大小
LEARNING_RATE = 3e-3         # 学习率
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-2

# ====== 模型参数 ======
DIM_INPUT = 25
DIM_HIDDEN = 12
DROPOUT = 0.2                # Dropout 比例

NUM_WORKERS = 4
RANDOM_SEED = 42

FEAT_SEQ_LEN = 3      # 提取eGeMaps时音频分段数量
SAMPLING_RATE = 16000  # 音频采样率

SECOND_LENGTH = 45          # 音频截取长度(秒)

class ModelConfig:
    """
    模型配置类（简化版）
    
    只包含 AD 二分类所需的参数
    """
    
    def __init__(
        self,
        dim_input: int = DIM_INPUT,
        dim_hidden: int = DIM_HIDDEN,
        dropout: float = DROPOUT,
    ):
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        
        # AD 任务标记
        self.do_ad = True
        
    def __repr__(self):
        return (f"ModelConfig(dim_input={self.dim_input}, "
                f"dim_hidden={self.dim_hidden}, "
                f"dropout={self.dropout})")


# 默认配置
DEFAULT_CONFIG = ModelConfig()

