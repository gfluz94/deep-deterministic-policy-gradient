import torch
import numpy as np

from ddpg import DDPG


class TestDDPG(object):
    def test_take_action(self):
        # OUTPUT
        state_space_dim = 5
        action_space_dim = 5
        hidden_layers = [5, 5]
        max_action_value = 10.0
        policy_update_freq = 2
        discount_factor = 0.99
        torch.manual_seed(0)
        agent = DDPG(
            state_space_dim=state_space_dim,
            action_space_dim=action_space_dim,
            hidden_layers=hidden_layers,
            max_action_value=max_action_value,
            policy_update_freq=policy_update_freq,
            discount_factor=discount_factor,
        )
        x = torch.ones(size=(1, state_space_dim))
        output = agent.take_action(x)

        # EXPECTED
        expected_output = np.array(
            [0.15068813, 0.84335005, -0.6132773, 0.6884175, -2.29879]
        )

        # ASSERT
        assert (np.float32(output) == np.float32(expected_output)).all()
