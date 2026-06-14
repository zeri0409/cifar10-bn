"""
Custom optimizers implemented without torch.optim internals.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class OptimizerBase:
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        self.params = [param for param in params if param.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, state: dict) -> None:
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]


class ManualSGD(OptimizerBase):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.velocity = [torch.zeros_like(param) for param in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for index, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param
            if self.momentum > 0.0:
                self.velocity[index].mul_(self.momentum).add_(grad)
                grad = self.velocity[index]
            param.add_(grad, alpha=-self.lr)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({
            "momentum": self.momentum,
            "velocity": [value.detach().cpu() for value in self.velocity],
        })
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.momentum = state["momentum"]
        self.velocity = [value.to(param.device) for value, param in zip(state["velocity"], self.params)]


class ManualAdam(OptimizerBase):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        for index, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param
            self.m[index].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            self.v[index].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            m_hat = self.m[index] / (1 - math.pow(self.beta1, self.t))
            v_hat = self.v[index] / (1 - math.pow(self.beta2, self.t))
            param.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "m": [value.detach().cpu() for value in self.m],
            "v": [value.detach().cpu() for value in self.v],
            "t": self.t,
        })
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]
        self.m = [value.to(param.device) for value, param in zip(state["m"], self.params)]
        self.v = [value.to(param.device) for value, param in zip(state["v"], self.params)]
        self.t = state["t"]
