import abc
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import *
import sys

from beartype import beartype
import datasets
import numpy as np
import more_itertools
import queue
import rich
import torch
import torch.nn as nn
import transformers
import wandb

TokenizerType = transformers.tokenization_utils_fast.PreTrainedTokenizerFast

class BaseEpsilonScheduler(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        pass

class LinearEpsilonScheduler(BaseEpsilonScheduler):
    def __init__(self, epsilon, num_steps):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.epoch = 0

    def __call__(self):
        self.epoch += 1
        epsilon = min(self.epsilon * (1 - self.epoch / self.num_epochs), 1)
        wandb.log({"epsilon": epsilon})
        wandb.log({"epsilon_num_steps": self.num_steps})
        return epsilon

class ConstantEpsilonScheduler(BaseEpsilonScheduler):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        epsilon = self.epsilon
        wandb.log({"epsilon": epsilon})
        return epsilon