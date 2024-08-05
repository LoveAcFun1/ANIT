from copy import deepcopy

import torch
import torch.nn as nn
from collections import OrderedDict


class AdvLearning:
    def __init__(self, model: nn.Module, gg: list[float] | None = None) -> None:
        self.model = model
        self.gg = gg  #
        self.embedgrad = OrderedDict()
        self.overallgrad = OrderedDict()

    def store_embedgrad(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():  # check all nn.Embedding
                if isinstance(module, nn.Embedding):
                    # self.embedgrad[name] = deepcopy(module.weight.grad)
                    self.embedgrad[name] = module.weight.grad.detach().clone()

    def store_overallgrad(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():  # iter all parameters
                # self.overallgrad[name] = param.grad
                self.overallgrad[name] = param.grad.detach().clone()

    def restore_embedgrad(self, reset: bool = True):
        assert self.embedgrad, "self.embedgrad is OrderDict(None)."
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Embedding):
                    module.weight.grad = deepcopy(self.embedgrad[name])

        if reset:
            # reset embedding gradient
            self.embedgrad = OrderedDict()

    def restore_overall(self, reset: bool = True, gg_optim: bool = False):
        assert self.overallgrad, "self.overallgrad is OrderDict(None)."

        with torch.no_grad():
            for name, param in self.model.named_parameters():  # iter all parameters
                param.grad = deepcopy(self.overallgrad[name])

        if reset:
            # reset overall gradient
            self.overallgrad = OrderedDict()