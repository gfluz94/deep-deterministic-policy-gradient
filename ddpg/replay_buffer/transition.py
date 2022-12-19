from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    """Class to store information regarding RL transitions."""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
