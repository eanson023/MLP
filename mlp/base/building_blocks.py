import copy
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp.utils import net_utils


class MultipleCriterions(nn.Module):
    """Container for multiple criterions.
    Since pytorch does not support ModuleDict(), we use ModuleList() to
    maintain multiple criterions.
    """

    def __init__(self, names=None, modules=None):
        super(MultipleCriterions, self).__init__()
        if names is not None:
            assert len(names) == len(modules)
        self.names = names if names is not None else []
        self.crits = nn.ModuleList(modules) if modules is not None else nn.ModuleList()
        self.name2crit = {}
        if names is not None:
            self.name2crit = {name: self.crits[i] for i, name in enumerate(names)}

    def forward(self, crit_inp, gts):
        self.loss = {}
        self.loss["total_loss"] = 0
        for name, crit in self.get_items():
            self.loss[name] = crit(crit_inp, gts)
            self.loss["total_loss"] += self.loss[name]
        return self.loss

    def add(self, name, crit):
        self.names.append(name)
        self.crits.append(crit)
        self.name2crit[name] = self.crits[-1]

    def get_items(self):
        return iter(zip(self.names, self.crits))

    def get_names(self):
        return self.names

    def get_crit_by_name(self, name):
        return self.name2crit[name]
