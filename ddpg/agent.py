from typing import List
import numpy as np
import torch

from ddpg.models import Actor, Critic
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
            action = self._actor(torch.Tensor(state)).cpu().data.numpy().reshape(-1)
        return action

    def train(
        self,
        replay_buffer: MemoryBuffer,
        batch_size: int,
        action_noise: float,
        maximum_noise_value: float,
        tau: float = 0.99,
    ) -> None:
        """Method that performs training of actor/critic models.

        Args:
            replay_buffer (MemoryBuffer): Memory buffer so that transitions are retrieved.
            batch_size (int): Batch size for transitions retrieval.
            action_noise (float): Standard deviation of Gaussian Noise applied to next action.
            maximum_noise_value (float): Maximum noise value allowed.
            tau (float): Learning rate for soft update of target actor/critic models.
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size=batch_size
        )
        self._actor = self._actor.train()
        for idx, state_, action_, reward_, next_state_, done_ in enumerate(
            zip(states, actions, rewards, next_states, dones)
        ):
            # Converting to tensors
            state = torch.Tensor(state_).to(device=_DEVICE)
            action = torch.Tensor(action_).to(device=_DEVICE)
            reward = torch.Tensor(reward_).to(device=_DEVICE)
            next_state = torch.Tensor(next_state_).to(device=_DEVICE)
            done = torch.Tensor(done_).to(device=_DEVICE)

            # Using actor target to predict next action on next state
            next_action = self._actor_target(next_state)

            # Add noise to action, so that we allow for exploration
            noise = torch.Tensor(next_action).data.normal_(mean=0, std=action_noise)
            noise.clamp_(min=-maximum_noise_value, max=maximum_noise_value)
            # Detach is critical to compute gradients correctly (only predicted)
            next_action = (
                (next_action + noise)
                .clamp(min=-self._max_action_value, max=self._max_action_value)
                .detach()
            )

            # Computing target Q(s, a)
            target_Q = reward + (
                1 - done
            ) * self._discount_factor * self._critic_target(next_state, next_action)

            # Predicted Q(s, a) with critic model
            predicted_Q = self._critic(state, action)

            # Backpropagation for critic network
            Q_loss = torch.nn.functional.mse_loss(predicted_Q, target_Q)
            self._critic_optimizer.zero_grad()
            Q_loss.backward()
            self._critic_optimizer.step()

            # Backpropagation for actor network
            loss_actor = -self._critic(state, self._actor(state)).mean()
            self._actor_optimizer.zero_grad()
            loss_actor.backward()
            self._actor_optimizer.step()

            if idx % self._policy_update_freq == 0:

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
