from pathlib import Path

# ====== Path configuration ======
# Get the absolute path of the config.py file
CONFIG_FILE = Path(__file__).resolve()
# ad_detection/train/config.py -> ad_detection/
PROJECT_ROOT = CONFIG_FILE.parent.parent

# ====== Training parameters ======
MAX_EPOCHS = 50              
BATCH_SIZE = 32              
LEARNING_RATE = 3e-3         #Best learning rate is 3e-3
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-2
TRAIN_SET_RATTIO = 0.8

# ====== Model parameters ======
#DIM_HIDDEN = 28, DROPOUT = 0.2, SECOND_LENGTH = 60  is the best configuration for XLSR
#DIM_HIDDEN = 14, DROPOUT = 0.2, SECOND_LENGTH = 60  is the best configuration for XLSR

DIM_INPUT = 25
DIM_HIDDEN = 14             # Network Hidden dimension
DROPOUT = 0.2                # Dropout ratio
RANDOM_SEEDS = [21, 42, 84, 168, 336]

# ====== eGeMAPS features extraction parameters ======
FEAT_SEQ_LEN = 10      # Number of audio segments when extracting eGeMaps

# ====== XLSR features extraction parameters ======
SECOND_LENGTH = 60     # XLSR extraction audio length (seconds)
XLSR_FEATURE_DIM = 1024  # XLSR feature dimension (output from XLSR-53 model)
XLSR_SEGMENT_LEN = 1  #XLSR segment length, average pooling = 1

# ====== Data loading parameters ======
NUM_WORKERS = 4
RANDOM_SEED = 42
SAMPLING_RATE = 16000  # Audio sampling rate

class ModelConfig:    
    def __init__(
        self,
        dim_input: int = DIM_INPUT,
        dim_hidden: int = DIM_HIDDEN,
        dropout: float = DROPOUT,
    ):
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        
        # AD task flag
        self.do_ad = True
        
    def __repr__(self):
        return (f"ModelConfig(dim_input={self.dim_input}, "
                f"dim_hidden={self.dim_hidden}, "
                f"dropout={self.dropout})")


# Default configuration
DEFAULT_CONFIG = ModelConfig()
