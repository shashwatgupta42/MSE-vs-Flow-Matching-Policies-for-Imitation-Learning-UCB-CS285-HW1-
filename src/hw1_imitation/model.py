"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.GELU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, action_dim * chunk_size))
        self.linear_gelu_stack = nn.Sequential(*layers)

        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        return self.linear_gelu_stack(state)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted_chunk = self.forward(state)
        return self.loss_fn(predicted_chunk, action_chunk.reshape(action_chunk.shape[0], -1))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        predicted_chunk = self.forward(state)
        return predicted_chunk.reshape(predicted_chunk.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim + action_dim * chunk_size + 1
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.GELU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, action_dim * chunk_size))
        self.linear_gelu_stack = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        A_t: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        # A_t is already 2D
        return self.linear_gelu_stack(torch.cat((state, A_t, timesteps), dim=1))

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = torch.empty(action_chunk.shape[0], 1, device=action_chunk.device).uniform_(0, 1)

        A_t0 = torch.randn(action_chunk.shape[0], self.action_dim*self.chunk_size, device=action_chunk.device)
        A_t = timesteps*action_chunk.reshape(action_chunk.shape[0], -1) + (1-timesteps)*A_t0

        v_t = self.forward(state, A_t, timesteps)
        loss = nn.functional.mse_loss(v_t, action_chunk.reshape(action_chunk.shape[0], -1) - A_t0)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        A_t = torch.randn(state.shape[0], self.action_dim*self.chunk_size, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = i * dt
            timesteps = torch.full((state.shape[0], 1), t, device=state.device)
            v_t = self.forward(state, A_t, timesteps)
            A_t += v_t * dt
        return A_t.reshape(A_t.shape[0], self.chunk_size, self.action_dim)



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
