from typing import Union
import torch

from ddpg.models.actor import Actor
from ddpg.models.critic import Critic
from ddpg.models.errors import ClassTypeNotAllowed


def save_model(model: Union[Actor, Critic], filepath: str) -> None:
    """Function to save Actor/Critic models.

    Args:
        model (Union[Actor, Critic]): Actor/Critic model to be saved.
        filepath (str): Name of the file where to save the model.
    """
    if not isinstance(model, Actor) and not isinstance(model, Critic):
        raise ClassTypeNotAllowed(
            "Only classes Actor/Critic allowed for method invocation!"
        )
    architecture = model.architecture()
    architecture["state_dict"] = model.state_dict()
    torch.save(architecture, filepath)


def load_model(filepath: str, model_class: object) -> Union[Actor, Critic]:
    """Function to load Actor/Critic models.

    Args:
        filepath (str): Name of the file where model is currently stored.
        model_class (object): Class of object to be retrieved [Actor, Critic]

    Returns:
        Union[Actor, Critic]: Retrieved model
    """
    if model_class not in [Actor, Critic]:
        raise ClassTypeNotAllowed(
            "Only classes Actor/Critic allowed for method invocation!"
        )
    architecture = torch.load(filepath)
    architecture["hidden_layers"] = list(
        map(int, architecture["hidden_layers"].split(" | "))
    )

    state_dict = architecture["state_dict"]
    del architecture["state_dict"]

    model = model_class(**architecture)
    model.load_state_dict(state_dict)

    return model
