from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple

@dataclass
class ModelConfig:
    pass


@dataclass
class SolverConfig:
    NAME: str = 'Adam'
    learning_rate: float = 5e-4  # 1e-3 originally.


@dataclass
class DefaultConfig:
    DEBUG: bool = False
    batch_size: int = 1
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
    num_epochs: int = 50
    optimizer: str = 'Adam'
    learning_rate: float = 5e-4  # 1e-3 originally.
    weight_decay: float = 1e-6  # 1e-3 originally.
    lr_step_size: int = 20
    LR_GAMMA: float = 0.5
    patch_size: Tuple[int] = (1024, 1024)
    SOLVER: SolverConfig = MISSING
    network: ModelConfig = MISSING


@dataclass
class UnetConfig(ModelConfig):
    AUGMENTATIONS = []
    use_classifier: bool = False
    classifier_gradient_multiplier: float = 0.1
    loss_name: str = 'basic'
    loss_top_k: float = 0.05
    loss_gamma: float = 2.
    num_base_filters: int = 64
    depth: int = 4
    dropout_probability: float = 0.1
