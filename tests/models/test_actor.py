import torch
import numpy as np

from ddpg.models import Actor


class TestActor(object):
    def test_actorInstantiated(self):
        # OUTPUT
        input_dim = 5
        output_dim = 5
        hidden_layers = [5, 5]
        max_output = 10.0
        actor = Actor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            max_output=max_output,
        )

        # EXPECTED
        architecture = {
            "input_dim": input_dim,
            "hidden_layers": "5 | 5",
            "output_dim": output_dim,
            "max_output": max_output,
        }

        # ASSERT
        assert architecture == actor.architecture()

    def test_actorOutputCorrect(self):
        # OUTPUT
        input_dim = 5
        output_dim = 5
        hidden_layers = [5, 5]
        max_output = 10.0
        torch.manual_seed(0)
        actor = Actor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            max_output=max_output,
        )
        x = torch.ones(size=(1, input_dim))
        output = actor(x)

        # EXPECTED
        expected_output = torch.Tensor(
            np.array([[0.15068813, 0.84335005, -0.6132773, 0.6884175, -2.29879]])
        )

        # ASSERT
        assert (output.detach().numpy() == expected_output.detach().numpy()).all()
