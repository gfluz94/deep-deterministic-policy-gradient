import numpy as np
import pytest

from ddpg.replay_buffer import MemoryBuffer
from ddpg.replay_buffer.transition import Transition
from ddpg.replay_buffer.errors import MemoryLowerThanBatchSize

class TestMemoryBuffer(object):

    def test_add_TransitionAdded(self, transition_not_done: Transition):
        # OUTPUT
        memory_buffer = MemoryBuffer()
        memory_buffer.add(transition=transition_not_done)

        # EXPECTED
        expected_output = 1

        # ASSERT
        assert len(memory_buffer) == expected_output

    def test_add_MaximumLengthExceeded(self, transition_not_done: Transition, transition_done: Transition):
        # OUTPUT
        memory_buffer = MemoryBuffer(max_length=1)
        memory_buffer.add(transition=transition_not_done)
        memory_buffer.add(transition=transition_done)

        # EXPECTED
        expected_output = 1

        # ASSERT
        assert len(memory_buffer) == expected_output


    def test_sample_ReturnedExpectedTransition(self, transition_not_done: Transition, arrays_not_done: np.ndarray):
        # OUTPUT
        memory_buffer = MemoryBuffer()
        memory_buffer.add(transition=transition_not_done)

        # EXPECTED
        expected_output = arrays_not_done

        # ASSERT
        for i in range(len(expected_output)):
            assert (memory_buffer.sample(batch_size=1)[i] == expected_output[i]).all()

    def test_sample_ReturnSecondTransition(self, transition_not_done: Transition, transition_done: Transition, arrays_done: np.ndarray):
        # OUTPUT
        memory_buffer = MemoryBuffer(max_length=1)
        memory_buffer.add(transition=transition_not_done)
        memory_buffer.add(transition=transition_done)

        # EXPECTED
        expected_output = arrays_done

        # ASSERT
        for i in range(len(expected_output)):
            assert (memory_buffer.sample(batch_size=1)[i] == expected_output[i]).all()

    
    def test_sample_RaiseExceptionBatchSizeHigherThanLength(self, transition_not_done: Transition):
        # OUTPUT
        memory_buffer = MemoryBuffer()
        memory_buffer.add(transition=transition_not_done)

        # ASSERT
        with pytest.raises(MemoryLowerThanBatchSize):
            memory_buffer.sample(batch_size=10)