from easydict import EasyDict as edict
import yaml
import numpy as np
import os

__C = edict()
cfg = __C

# Local/remote config option
__C.LOCAL = False
__C.DEBUG = False

# Directory setup
__C.INPUT_DIR = '../experiments'
__C.EXP_DIR = '../experiments'
__C.DATA_SOURCE = '../dataset/share'
__C.LIST_DIR = '../dataset/share'

# Multi-GPU/FP16 settings
__C.APEX = -1  # -1 for APEX disabled, i >=0 for Oi optimization
__C.MULTIGPU = 0  # 0 for single GPU, 1 for DataParallel, 2 for DistributedDataParallel
__C.DELAY_REDUCE = False

# Filename params


# Network config
__C.BATCH_SZ = 1  # Images in a batch

__C.UNET = edict()
__C.UNET.OUTPUT_SHAPE = (1, 512, 512)
__C.UNET.NUM_CLS = 1
__C.UNET.DOWN_BLOCKS = [(3, 0, 32), (3, 0, 32), (3, 0, 64), (1, 2, 64), (1, 2, 128), (1, 2, 128), (0, 4, 256, False)]
__C.UNET.UP_BLOCKS = [(4, 256, False), (4, 128), (4, 128), (3, 64), (3, 64), (3, 64), (3, 64)]
__C.UNET.BOTTLENECK_CH = 1024
__C.UNET.AUG_LST = ['RandomGammaTransform', 'RandomShiftTransform', 'RandomZoomTransform']
__C.UNET.FLIP_AXIS = [-1, 0]  # Axis which can be flipped
__C.UNET.FLIP_MASK = [True, False]  # Flip mask in given axis?
__C.UNET.NUM_AUG = 3
__C.UNET.NOISE_STD = 0.05
__C.UNET.GAMMA_RANGE = 0.4
__C.UNET.AUG_INTERP = 'LINEAR'  # LINEAR or NEAREST


# Training and debug output options
__C.HARD_MINING_FREQ = 0
__C.POS_SAMPLE_WEIGHT = 0.5
__C.REPORT_INTERVAL = 1
__C.GRAD_STEPS = 1
__C.GRAD_CLIP = 0
__C.TOPK = 0.1
__C.LOSS = 'topkbce'
__C.N_EPOCHS = 50
__C.OPTIMIZER = 'Adam'
__C.STARTER_LR = 5e-4  # 1e-3 originally.
__C.WEIGHT_DECAY = 1e-6  # 1e-3 originally.
__C.LR_STEP_SIZE = 20
__C.LR_GAMMA = 0.5
__C.DRAW_PLOT = False
__C.SAVE_PLOT = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key.'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)
