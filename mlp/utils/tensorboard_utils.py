# Pytorch tensorflow summary for using tensorboard.
# This utility is motivated by https://github.com/lanpa/tensorboard-pytorch.

from tensorboardX import SummaryWriter


class PytorchSummary(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir, flush_secs=5)
        self.summary_dict = {}
        self.input_dict = {}

    def add_scalar(self, name, value, global_step=0):
        writer = self.writer
        writer.add_scalar(name, value, global_step=global_step)

    def add_histogram(self, name, values, global_step=0):
        writer = self.writer
        writer.add_histogram(name, values, global_step=global_step)
