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
        output = actor(x).detach().numpy()

        # EXPECTED
        expected_output = np.array(
            [[0.15068813, 0.84335005, -0.6132773, 0.6884175, -2.29879]],
            dtype=np.float32,
        )

        # ASSERT
        assert (np.round(output, 5) == np.round(expected_output, 5)).all()
