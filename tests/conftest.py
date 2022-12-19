"""
Module that contains objects used as inputs for test scenarios.
"""

from typing import Tuple
import pytest
import numpy as np

from ddpg.replay_buffer.transition import Transition


@pytest.fixture(scope="module")
def transition_not_done() -> Transition:
    """Transition example for testing purposes, with `done` set to False.

    Returns:
        Transition: Mocked transition example for testing.
    """
    return Transition(
        state=np.array([0.5, 0.25, 0.5]),
        action=np.array([1.0, -1.0, 1.0]),
        reward=0.25,
        next_state=np.array([1.5, 0.0, 0.80]),
        done=False,
    )


@pytest.fixture(scope="module")
def arrays_not_done() -> Tuple[np.ndarray]:
    """Numpy arrays expected to be returned from sampling transitions.

    Returns:
        Tuple[np.ndarray]: Mocked transition example for testing.
    """
    return (
        np.array([[0.5, 0.25, 0.5]]),
        np.array([[1.0, -1.0, 1.0]]),
        np.array([[0.25]]),
        np.array([[1.5, 0.0, 0.80]]),
        np.array([[0.0]]),
    )


@pytest.fixture(scope="module")
def transition_done() -> Transition:
    """Transition example for testing purposes, with `done` set to True.

    Returns:
        Transition: Mocked transition example for testing.
    """
    return Transition(
        state=np.array([0.5, 0.25, 0.5]),
        action=np.array([1.0, -1.0, 1.0]),
        reward=0.25,
        next_state=np.array([1.5, 0.0, 0.80]),
        done=True,
    )


@pytest.fixture(scope="module")
def arrays_done() -> Tuple[np.ndarray]:
    """Numpy arrays expected to be returned from sampling transitions.

    Returns:
        Tuple[np.ndarray]: Mocked transition example for testing.
    """
    return (
        np.array([[0.5, 0.25, 0.5]]),
        np.array([[1.0, -1.0, 1.0]]),
        np.array([[0.25]]),
        np.array([[1.5, 0.0, 0.80]]),
        np.array([[1.0]]),
    )
