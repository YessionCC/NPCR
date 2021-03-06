from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.FEATURE_DIM = 20

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the original image
_C.INPUT.SIZE_RAW = [1920,1200]
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [720,450]
# Size of the image during test
_C.INPUT.SIZE_TEST = [720,450]
#
_C.INPUT.LAYER_NUM = 1
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.NEAR_FAR_SIZE = [1000, 3600, 1.5]
_C.INPUT.USE_RGB = True

_C.INPUT.USE_DIR = "MAPS"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = "./"
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.FRAME_NUM = 20
_C.DATASETS.MASK = False
_C.DATASETS.CENTER = []
_C.DATASETS.SHIFT = 0
_C.DATASETS.MAXRATION = 0.0
_C.DATASETS.ROTATION = 0.0

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 10

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 20

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 1

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""

_C.WEIGHTS = CN()
_C.WEIGHTS.REFINE_WEIGHTS=[]
_C.WEIGHTS.RENDER_WEIGHTS=[]
_C.WEIGHTS.COLOR_THR=0
_C.WEIGHTS.DEPTH_THR = 10.0
_C.BACKGROUND_GRAY=255
_C.LOAD_EPOCH=0
_C.RESUME=False
