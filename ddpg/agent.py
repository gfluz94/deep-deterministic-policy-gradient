from typing import List
import numpy as np
import torch

from ddpg.models import Actor, Critic, save_model, load_model
from ddpg.replay_buffer import MemoryBuffer

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    """Deep Deterministic Policy Gradient agent for RL in MuJoCo environment.

    Parameters:
        state_space_dim (int): State space dimension.
        action_space_dim (int): Action space dimension.
        hidden_layers (List[int]): Hidden layers' units.
        max_action_value (float): Maximum action value allowed in environment.
        policy_update_freq (int): Frequency with which we update target actor/critic models.
        discount_factor (float): Discount factor for estimating target Q-values.
    """

    def __init__(
        self,
        state_space_dim: int,
        action_space_dim: int,
        hidden_layers: List[int],
        max_action_value: float,
        policy_update_freq: int,
        discount_factor: float,
    ) -> None:
        """Constructor method of DDPG object.

        Args:
            state_space_dim (int): State space dimension.
            action_space_dim (int): Action space dimension.
            hidden_layers (List[int]): Hidden layers' units.
            max_action_value (float): Maximum action value allowed in environment.
            policy_update_freq (int): Frequency with which we update target actor/critic models.
            discount_factor (float): Discount factor for estimating target Q-values.

        Returns:
           (None)
        """
        self._state_space_dim = state_space_dim
        self._action_space_dim = action_space_dim
        self._hidden_layers = hidden_layers
        self._max_action_value = max_action_value
        self._policy_update_freq = policy_update_freq
        self._discount_factor = discount_factor

        # Defining actor and actor target with same weight initialization
        self._actor = Actor(
            input_dim=self._state_space_dim,
            output_dim=self._action_space_dim,
            hidden_layers=self._hidden_layers,
            max_output=self._max_action_value,
        ).to(device=_DEVICE)
        self._actor_target = Actor(
            input_dim=self._state_space_dim,
            output_dim=self._action_space_dim,
            hidden_layers=self._hidden_layers,
            max_output=self._max_action_value,
        ).to(device=_DEVICE)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(params=self._actor.parameters())

        # Defining critic and critic target with same weight initialization
        self._critic = Critic(
            input_dim=self._state_space_dim + self._action_space_dim,
            hidden_layers=self._hidden_layers,
        ).to(device=_DEVICE)
        self._critic_target = Critic(
            input_dim=self._state_space_dim + self._action_space_dim,
            hidden_layers=self._hidden_layers,
        ).to(device=_DEVICE)
        self._critic_target.load_state_dict(self._critic.state_dict())
        self._critic_optimizer = torch.optim.Adam(params=self._critic.parameters())

    @property
    def state_space_dim(self) -> int:
        return self._state_space_dim

    @property
    def action_space_dim(self) -> int:
        return self._action_space_dim

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    @property
    def max_action_value(self) -> int:
        return self._max_action_value

    @property
    def policy_update_freq(self) -> int:
        return self._policy_update_freq

    @property
    def discount_factor(self) -> int:
        return self._discount_factor

    def take_action(self, state: np.ndarray) -> np.ndarray:
        """Method to allow for exploitation - DDPG agent chooses best action, according to policy.

        Args:
            state (np.ndarray): Current environment's state.

        Returns:
            np.ndarray: Action to be carried out in environment.
        """
        self._actor = self._actor.eval()
        with torch.no_grad():
            action = (
                self._actor(torch.Tensor(state.reshape(1, -1)))
                .cpu()
                .data.numpy()
                .reshape(-1)
            )
        return action

    def save(self, filepreffix: str) -> None:
        save_model(self._actor, f"{filepreffix}_actor.pt")
        save_model(self._critic, f"{filepreffix}_critic.pt")

    def load(self, filepreffix: str) -> None:
        self._actor = load_model(f"{filepreffix}_actor.pt", model_class=Actor)
        self._critic = load_model(f"{filepreffix}_critic.pt", model_class=Critic)

    def train(
        self,
        replay_buffer: MemoryBuffer,
        batch_size: int,
        episode_length: int,
        tau: float = 0.99,
    ) -> None:
        """Method that performs training of actor/critic models.

        Args:
            replay_buffer (MemoryBuffer): Memory buffer so that transitions are retrieved.
            batch_size (int): Batch size for transitions retrieval.
            episode_length (int): Number of episodes played.
            tau (float): Learning rate for soft update of target actor/critic models.
        """
        self._actor = self._actor.train()
        for idx in range(episode_length):
            # Sample from batch
            states, actions, rewards, next_states, dones = replay_buffer.sample(
                batch_size=batch_size
            )
            # Converting to tensors
            state = torch.Tensor(states).to(device=_DEVICE)
            action = torch.Tensor(actions).to(device=_DEVICE)
            reward = torch.Tensor(rewards).to(device=_DEVICE)
            next_state = torch.Tensor(next_states).to(device=_DEVICE)
            done = torch.Tensor(dones).to(device=_DEVICE)

            # Using actor target to predict next action on next state
            next_action = self._actor_target(next_state)

            # Computing target Q(s, a)
            target_Q = (
                reward
                + (
                    (1 - done)
                    * self._discount_factor
                    * self._critic_target(next_state, next_action)
                ).detach()
            )

            # Predicted Q(s, a) with critic model
            predicted_Q = self._critic(state, action)

            # Backpropagation for critic network
            Q_loss = torch.nn.functional.mse_loss(predicted_Q, target_Q)
            self._critic_optimizer.zero_grad()
            Q_loss.backward()
            self._critic_optimizer.step()

            if idx % self._policy_update_freq == 0:

                # Backpropagation for actor network
                loss_actor = -self._critic.Q(state, self._actor(state)).mean()
                self._actor_optimizer.zero_grad()
                loss_actor.backward()
                self._actor_optimizer.step()

                # Update actor target network
                for param, target_param in zip(
                    self._actor.parameters(), self._actor_target.parameters()
                ):
                    target_param.data.copy_(
                        target_param.data * tau + param.data * (1 - tau)
                    )

                # Update critic target network
                for param, target_param in zip(
                    self._critic.parameters(), self._critic_target.parameters()
                ):
                    target_param.data.copy_(
                        target_param.data * tau + param.data * (1 - tau)
                    )
