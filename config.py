DATA_DIR = 'data'
RESULTS_DIR = 'results'
CHECKPOINTS_DIR = 'results/checkpoints'
PLOTS_DIR = 'results/plots'
LOGS_DIR = 'results/logs'

MODELS = ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small']
INPUT_SIZES = [128, 224, 256, 320]
AUGMENTATIONS = [False, True]

DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_EPOCHS = 30
DEFAULT_INPUT_SIZE = 224

# Hyperparameter search space
LR_OPTIONS = [0.001, 0.0005, 0.0001]
BATCH_SIZE_OPTIONS = [16, 32]
OPTIMIZER_OPTIONS = ['adam', 'adamw']

EARLY_STOPPING_PATIENCE = 10
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
