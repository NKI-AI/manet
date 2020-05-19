from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class ModelConfig:
    pass


@dataclass
class SolverConfig:
    NAME: str = 'Adam'
    STARTER_LR: float = 5e-4  # 1e-3 originally.


@dataclass
class DefaultConfig:
    DEBUG: bool = False
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
    SOLVER: SolverConfig = MISSING
    NETWORK: ModelConfig = MISSING


@dataclass
class UnetConfig(ModelConfig):
    AUGMENTATIONS = []
    USE_CLASSIFIER: bool = False
    CLASSIFIER_GRADIENT_MULT: float = 0.1
    LOSS_NAME: int = 'basic'
