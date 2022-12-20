import torch
import numpy as np

from ddpg.models import Critic


class TestCritic(object):
    def test_criticInstantiated(self):
        # OUTPUT
        input_dim = 5
        hidden_layers = [5, 5]
        critic = Critic(input_dim=input_dim, hidden_layers=hidden_layers)

        # EXPECTED
        architecture = {
            "input_dim": input_dim,
            "hidden_layers": "5 | 5",
        }

        # ASSERT
        assert architecture == critic.architecture()

    def test_criticOutputCorrect(self):
        # OUTPUT
        input_dim = 10
        hidden_layers = [5, 5]
        torch.manual_seed(0)
        critic = Critic(input_dim=input_dim, hidden_layers=hidden_layers)
        x1 = torch.ones(size=(1, input_dim // 2))
        x2 = torch.ones(size=(1, input_dim // 2))
        output = critic(x1, x2)

        # EXPECTED
        expected_output = torch.Tensor(np.array([[0.0639877]]))

        # ASSERT
        assert (output.detach().numpy() == expected_output.detach().numpy()).all()
