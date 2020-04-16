from dataclasses import dataclass

@dataclass
class DefaultConfig:
    BATCH_SIZE: int = 1
    APEX: int = -1
    MULTIGPU: int = 0
    # DELAY_REDUCE: bool = False
    REPORT_INTERVAL: int = 5

    # Training and debug output options
    HARD_MINING_FREQ: int = 0
    # POS_SAMPLE_WEIGHT = 0.5
    REPORT_INTERVAL: int = 1
    GRAD_STEPS: int = 1
    GRAD_CLIP: float = 0
    N_EPOCHS: int = 50
    OPTIMIZER: str = 'Adam'
    STARTER_LR: float = 5e-4  # 1e-3 originally.
    WEIGHT_DECAY: float = 1e-6  # 1e-3 originally.
    LR_STEP_SIZE: int = 20
    LR_GAMMA: float = 0.5



#
# # Network config
# __C.BATCH_SZ = 1  # Images in a batch
#
# __C.UNET = edict()
# __C.UNET.OUTPUT_SHAPE = (1, 512, 512)
# __C.UNET.NUM_CLS = 1
# __C.UNET.DOWN_BLOCKS = [(3, 0, 32), (3, 0, 32), (3, 0, 64), (1, 2, 64), (1, 2, 128), (1, 2, 128), (0, 4, 256, False)]
# __C.UNET.UP_BLOCKS = [(4, 256, False), (4, 128), (4, 128), (3, 64), (3, 64), (3, 64), (3, 64)]
# __C.UNET.BOTTLENECK_CH = 1024
# __C.UNET.AUG_LST = ['RandomGammaTransform', 'RandomShiftTransform', 'RandomZoomTransform']
# __C.UNET.FLIP_AXIS = [-1, 0]  # Axis which can be flipped
# __C.UNET.FLIP_MASK = [True, False]  # Flip mask in given axis?
# __C.UNET.NUM_AUG = 3
# __C.UNET.NOISE_STD = 0.05
# __C.UNET.GAMMA_RANGE = 0.4
# __C.UNET.AUG_INTERP = 'LINEAR'  # LINEAR or NEAREST
# __C.UNET.USE_CLASSIFIER = False
#

