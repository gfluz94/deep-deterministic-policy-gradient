__all__ = ["MemoryBuffer"]

from typing import Tuple
import numpy as np

from ddpg.replay_buffer.transition import Transition


class MemoryBuffer(object):
    """Experience replay buffer where we save transitions from past episodes.
    Training occurs on top of samples from the experience.

    Parameters:
        max_length (int): Maximum length of transitions stored in the memory
    """

    def __init__(self, max_length: int = 1e4) -> None:
        """Constructor method of MemoryBuffer object.

        Args:
            max_length (int): Maximum length of transitions stored in the memory. Defaults to 1e4.

        Returns:
           (None)
        """
        self._max_length = max_length
        self._transitions = []

    @property
    def max_length(self) -> int:
        return self._max_length

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: Transition) -> None:
        """Method that adds new transition to buffer.

        Args:
            transition (Transition): Transition data class containing (s, a, r, s', done) information.

        Returns:
            (None)
        """
        if self.__len__() >= self._max_length:
            self._transitions = [transition] + self._transitions[1:]
        else:
            self._transitions.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray]:
        """Method that returns sample transitions from memory.

        Args:
            batch_size (int): Size of sample batch.

        Returns:
            Tuple[np.ndarray]: Numpy arrays of states, actions, rewards, next states and dones.
        """
        indices = np.random.randint(low=0, high=self.__len__(), size=batch_size)
        states, actions, rewards, next_states, dones = ([] for _ in range(len(Transition.__annotations__)))
        for idx in indices:
            transition = self._transitions[idx]
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(1.0*transition.done)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1)
        )