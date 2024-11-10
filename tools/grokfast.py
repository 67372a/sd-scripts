# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grokfast.py

import math
from collections import deque
from typing import Dict, Literal, Optional

import torch
from torch import nn
from tools.stochastic_copy import copy_stochastic_

FILTER_TYPE = Literal['mean', 'sum']

class Gradfilter_ma:

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 25,
        lamb: float = 5.0,
        filter_type: FILTER_TYPE = 'mean',
        warmup_steps: int = 25,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.window_size = window_size if window_size is not None else 25
        self.lamb = lamb if lamb is not None else 5.0
        self.warmup_steps = warmup_steps if warmup_steps is not None else window_size
        self.filter_type = filter_type
        self.dtype = dtype
        self.step = 0
        self.grads = None

    def __str__(self) -> str:
        return 'Gradfilter_ma'
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "model"
        }
    
    @torch.no_grad()
    def filter(
        self
    ):
        r"""Grokfast-MA.

        Example:
        -------
            Here's an example::

                loss.backwards()  # Calculate the gradients.

                gradfilter_ma.filter()

                optimizer.step()  # Call the optimizer.

        :param model: nn.Module. model that contains every trainable parameters.
        :param window_size: int. the width of the filter window. additional memory requirements increases linearly with
            respect to the windows size.
        :param lamb: float. amplifying factor hyperparameter of the filter.
        :param filter_type: FILTER_TYPE. aggregation method for the running queue.
        :param warmup_steps: int. warmup filter instead of applying at full effect.
        :param dtype: dtype. dtype to store queued gradients as.
        """
        self.step += 1

        if self.grads is None:
            self.grads = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.grads[n] = None

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                if self.grads[n] is None:
                    self.grads[n] = deque(maxlen=self.window_size)

                if p.dtype in {torch.float16, torch.bfloat16} and p.grad.dtype == torch.float32:
                    temp_grad = torch.zeros_like(p)
                    copy_stochastic_(temp_grad, p.grad)
                else:
                    temp_grad = p.grad.detach().clone()

                self.grads[n].append(temp_grad)

                if self.warmup_steps <= self.step:
                    if self.filter_type == 'mean':
                        avg = sum({stored_grad.to(torch.float32) for stored_grad in self.grads[n]}) / len(self.grads[n])
                    elif self.filter_type == 'sum':
                        avg = sum({stored_grad.to(torch.float32) for stored_grad in self.grads[n]})
                    else:
                        raise ValueError(f'not supported filter_type {self.filter_type}')

                    p.grad.add_(avg, alpha=self.lamb)

class Gradfilter_ema:

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.98,
        lamb: float = 2.0,
        warmup_steps: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.alpha = alpha if alpha is not None else 0.98
        self.lamb = lamb if lamb is not None else 2.0
        self.warmup_steps = warmup_steps if warmup_steps is not None else 0
        self.dtype = dtype
        self.step = 0
        self.grads = None

    def __str__(self) -> str:
        return 'Gradfilter_ema'
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "model"
        }

    @torch.no_grad()
    def filter(
        self,
    ):
        r"""Grokfast.

        Example:
        -------
            Here's an example::

                loss.backwards()  # Calculate the gradients.

                gradfilter_ema.filter()

                optimizer.step()  # Call the optimizer.

        :param model: nn.Module. model that contains every trainable parameters.
        :param alpha: int. momentum hyperparameter of the EMA.
        :param lamb: float. amplifying factor hyperparameter of the filter.
        :param dtype: dtype. dtype to store ema as.
        :param warmup_steps: int. warmup filter instead of applying at full effect.
        """

        self.step += 1

        if self.grads is None:
            self.grads = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.grads[n] = None

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if self.grads[n] is None:
                        self.grads[n] = torch.zeros_like(p)

                    grad = p.grad
                    ema = self.grads[n]

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        ema = ema.to(torch.float32)

                    ema.mul_(self.alpha).add_(grad, alpha=1.0 - self.alpha)

                    grad.add_(ema, alpha=self.lamb * min(1.0, (self.step / max(1, self.warmup_steps))))

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(self.grads[n], ema)