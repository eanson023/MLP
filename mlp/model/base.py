import copy
import json
import os
import torch
import torch.nn as nn
from abc import abstractmethod
from collections import OrderedDict
from omegaconf import DictConfig

from mlp.utils import accumulator, eval_utils
from mlp.utils.tensorboard_utils import PytorchSummary


class AbstractNetwork(nn.Module):
    def __init__(self, optim_config: DictConfig, working_dir: str,
                 dataset="babel", use_gpu: bool = False, logger=None, verbose=False):
        super(AbstractNetwork, self).__init__()  # Must call super __init__()

        # create internal variables
        self.optimizer = None
        self.models_to_update = None
        self.training_mode = True
        self.best_score = None
        self.use_tf_summary = False
        self.it = 0  # it: iteration
        self.total_step = 1  # total step
        self.total_epoch = -1

        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.working_dir = working_dir

        # save configuration for later network reproduction
        self.optim_config = optim_config

        self.dataset = dataset

        # prepare logging
        self.log = print
        if logger is not None:
            self.log = logger.info
        # self.log(json.dumps(config, indent=2))
        self.verbose = verbose

    """ methods for forward/backward """

    @abstractmethod
    def forward(self, net_inps, mode=None):
        """ Forward network
        Args:
            net_inps: inputs for network; dict()
        Returns:
            net_outs: dictionary including inputs for criterion, etc
        """
        pass

    def loss_fn(self, crit_inp, gts, count_loss=True):
        """ Compute loss
        Args:
            crit_inp: inputs for criterion which is outputs from forward(); dict()
            gts: ground truth
            count_loss: flag of accumulating loss or not (training or inference)
        Returns:
            loss: results of self.criterion; dict()
        """
        self.loss = self.criterion(crit_inp, gts)
        for name in self.loss.keys():
            self.status[name] = self.loss[name].item()
        if count_loss:
            for name in self.loss.keys():
                self.counters[name].add(self.status[name], 1)
        return self.loss

    def update(self, loss):
        """ Update the network
        Args:
            loss: loss to train the network; dict()
        """

        self.it = self.it + 1
        # initialize optimizer
        if self.optimizer == None:
            self.create_optimizer() 

        total_loss = loss["total_loss"]
        self.optimizer.zero_grad()  # set gradients as zero before update
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # self.optimizer.zero_grad()  # set gradients as zero before updating the network

    def forward_update(self, net_inps, gts):
        """ Forward and update the network at the same time
        Args:
            net_inps: inputs for network; dict()
            gts: ground truth; dict()
        Returns:
            {loss, net_output}: two items of dictionary
                - loss: results from self.criterion(); dict()
                - net_output: output from self.forward(); dict()
        """

        net_out = self.forward(net_inps)
        loss = self.loss_fn(net_out, gts, count_loss=True)
        self.update(loss)
        return {"loss": loss, "net_output": net_out}

    def compute_loss(self, net_inps, gts, mode="eval"):
        """ Compute loss and network's output at once
        Args:
            net_inps: inputs for network; dict()
            gts: ground truth; dict()
        Returns:
            {loss, net_output}: two items of dictionary
                - loss: results from self.criterion(); dict()
                - net_output: first output from self.forward(); dict()
        """
        net_out = self.forward(net_inps, mode)
        loss = self.loss_fn(net_out, gts, count_loss=True)
        return {"loss": loss, "net_output": net_out}

    def forward_only(self, net_inps, mode="eval"):
        """ network's output at once
        Args:
            net_inps: inputs for network; dict()
            gts: ground truth; dict()
        Returns:
            {loss, net_output}: two items of dictionary
                - loss: results from self.criterion(); dict()
                - net_output: first output from self.forward(); dict()
        """
        net_out = self.forward(net_inps, mode)
        return {"net_output": net_out}

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def create_optimizer(self):
        """ Create optimizer for training phase
        Currently supported optimizer list: [SGD, Adam]
        Args:
            lr: learning rate; int
        """

        # setting optimizer
        lr = self.optim_config["learning_rate"]
        opt_type = self.optim_config["optimizer_type"]
        self.clip_norm = self.optim_config["clip_norm"]
        if opt_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self.get_parameters(), lr=lr,
                momentum=self.optim_config["momentum"],
                weight_decay=self.optim_config["weight_decay"])
        elif opt_type == "AdamW":
            betas = self.optim_config["betas"]
            # no decay for parameters of layer norm and bias
            no_decay = ['bias', 'layer_norm', 'LayerNorm']
            weight_decay = self.optim_config["weight_decay"]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.get_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in self.get_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif opt_type == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.get_parameters(), lr=lr)
        elif opt_type == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.get_parameters(), lr=lr)
        else:
            raise NotImplementedError(
                "Not supported optimizer [{}]".format(opt_type))

        # setting scheduler
        self.scheduler = None
        scheduler_type = self.optim_config["scheduler_type"]
        decay_factor = self.optim_config["decay_factor"]
        decay_step = self.optim_config["decay_step"]
        if scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, int(decay_step / self.total_epoch * self.total_step), decay_factor)
        elif scheduler_type == "multistep":
            milestones = self.optim_config["milestones"]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones, decay_factor)
        elif scheduler_type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, decay_factor)
        elif scheduler_type == "lambda":
            def lambda1(it):
                return it // int(decay_step / self.total_epoch * self.total_step)

            def lambda2(it):
                return decay_factor ** it

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, [lambda1, lambda2])
        elif scheduler_type == "warmup":
            raise NotImplementedError()

    @abstractmethod
    def _build_network(self):
        pass

    def _build_evaluator(self):
        self.evaluator = eval_utils.get_evaluator(self.dataset)

    @abstractmethod
    def prepare_batch(self, batch):
        """ Prepare batch to be used for network
        e.g., shipping batch to gpu
        Args:
            batch: batch data; dict()
        Returns:
            net_inps: network inputs; dict()
            gts: ground-truths; dict()
        """
        pass

    @abstractmethod
    def apply_curriculum_learning(self):
        pass

    def save_results(self, prefix, mode="Train"):
        pass

    def _get_score(self):
        """ Method for status (losses, metrics)
        """
        return sum([float(metric.split("-")[1])*self.counters[metric].get_average() for metric in self.evaluator.get_metrics()])

    def renew_best_score(self):
        cur_score = self._get_score()
        if (self.best_score is None) or (cur_score > self.best_score):
            self.best_score = cur_score
            self.log("Iteration {}: New best score {:4f}".format(
                self.it, self.best_score))
            return True
        self.log("Iteration {}: Current score {:4f}".format(self.it, cur_score))
        self.log("Iteration {}: Current best score {:4f}".format(
            self.it, self.best_score))
        return False

    def reset_status(self, init_reset=False):
        """ Reset (initialize) metric scores or losses (status).
        """
        if init_reset:
            self.status = OrderedDict()
            self.status["total_loss"] = 0
            for k, v in self.criterion.get_items():
                self.status[k] = 0
            for k in self.evaluator.metrics:
                self.status[k] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0

    @abstractmethod
    def compute_status(self, net_outs, gts, mode="Train"):
        """ Compute metric scores or losses (status).
            You may need to implement this method.
        Args:
            net_outs: output of network.
            gts: ground-truth
        """
        pass

    def _get_print_list(self, mode):
        if mode == "Train":
            print_list = copy.deepcopy(self.criterion.get_names())
            print_list.append("total_loss")
        else:
            print_list = copy.deepcopy(self.evaluator.metrics)
        return print_list

    def print_status(self, enter_every=3):
        """ Print current metric scores or losses (status).
            You are encouraged to implement this method.
        Args:
            epoch: current epoch
        """
        val_list = self._get_print_list("Train")
        # print status information
        txt = "Step {} ".format(self.it)
        for i, (k) in enumerate(val_list):
            v = self.status[k]
            if (i + 1) % enter_every == 0:
                txt += "{} = {:.4f}, ".format(k, float(v))
                self.log(txt)
                txt = ""
            else:
                txt += "{} = {:.4f}, ".format(k, float(v))
        if len(txt) > 0:
            self.log(txt)

    def _create_counters(self):
        """ methods for counters """
        self.counters = OrderedDict()
        self.counters["total_loss"] = accumulator.Accumulator("total_loss")
        for k, v in self.criterion.get_items():
            self.counters[k] = accumulator.Accumulator(k)
        for k in self.evaluator.metrics:
            self.counters[k] = accumulator.Accumulator(k)

    def reset_counters(self):
        for k, v in self.counters.items():
            v.reset()

    def print_counters_info(self, logger, epoch, mode="Train"):
        val_list = self._get_print_list(mode)
        txt = "[{}] {} epoch {} iter".format(mode, epoch, self.it)
        for k in val_list:
            v = self.counters[k]
            txt += ", {} = {:.4f}".format(v.get_name(), v.get_average())
        if logger:
            logger.info(txt)
        else:
            self.log(txt)

        if self.use_tf_summary:
            self.write_counter_summary(epoch, mode)

        # reset counters
        self.reset_counters()

    """ methods for checkpoint """

    def load_checkpoint(self, ckpt_path, load_crit=False):
        """ Load checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path; str
        """
        epoch, it = 0, 0
        self.log("Checkpoint is loaded from {}".format(ckpt_path))
        model_state_dict = torch.load(
            ckpt_path, map_location=lambda storage, loc: storage)
        self.log("[{}] are in checkpoint".format(
            "|".join(model_state_dict.keys())))
        for m in model_state_dict.keys():
            if (not load_crit) and (m == "criterion"):
                continue
            if m == "epoch":
                epoch = model_state_dict[m]
                continue
            if m == "it":
                it = model_state_dict[m]
                continue
            if m in self.model_list:
                self.log("Initializing [{}] from checkpoint".format(m))
                self[m].load_state_dict(model_state_dict[m])
            else:
                self.log("{} is not in {}".format(
                    m, "|".join(self.model_list)))
        return epoch, it

    def save_checkpoint(self, ckpt_path, epoch, save_crit=False):
        """ Save checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path
        """
        model_state_dict = {m: self[m].state_dict()
                            for m in self.model_list if m != "criterion"}
        if save_crit:
            model_state_dict["criterion"] = self["criterion"].state_dict()
        model_state_dict["epoch"] = epoch
        model_state_dict["it"] = self.it
        torch.save(model_state_dict, ckpt_path)

        self.log("Checkpoint [{}] is saved in {}".format(
            " | ".join(model_state_dict.keys()), ckpt_path))

    """ methods for tensorboard """

    def create_tensorboard_summary(self, tensorboard_dir):
        self.use_tf_summary = True
        self.summary = PytorchSummary(tensorboard_dir)

    def set_tensorboard_summary(self, summary):
        self.use_tf_summary = True
        self.summary = summary

    def write_counter_summary(self, epoch, mode):
        for k, v in self.counters.items():
            self.summary.add_scalar(mode + '/counters/' + v.get_name(),
                                    v.get_average(), global_step=epoch)

    """ wrapper methods of nn.Modules """

    def get_parameters(self):
        if self.models_to_update is None:
            for name, param in self.named_parameters():
                yield name, param
        else:
            for m in self.models_to_update:
                if isinstance(self[m], dict):
                    for k, v in self[m].items():
                        for name, param in v.named_parameters():
                            yield name, param
                else:
                    for name, param in self[m].named_parameters():
                        yield name, param

    def cpu_mode(self):
        self.log("Setting cpu() for [{}]".format(" | ".join(self.model_list)))
        self.cpu()

    def gpu_mode(self):
        # cudnn.benchmark = False
        if torch.cuda.is_available():
            self.log("Setting gpu() for [{}]".format(
                " | ".join(self.model_list)))
            self.cuda()
        else:
            raise NotImplementedError("Available GPU not exists")

    def train_mode(self):
        self.train()
        self.training_mode = True
        if self.verbose:
            self.log("Setting train() for [{}]".format(
                " | ".join(self.model_list)))

    def eval_mode(self):
        self.eval()
        self.training_mode = False
        if self.verbose:
            self.log("Setting eval() for [{}]".format(
                " | ".join(self.model_list)))

    """ related to configuration or dataset """

    def bring_dataset_info(self, dset):
        print("You would need to implement 'bring_dataset_info'")
        pass

    def model_specific_config_update(self, config):
        print("You would need to implement 'model_specific_config_update'")
        return config

    @staticmethod
    def dataset_specific_config_update(config, dset):
        print("You would need to implement 'dataset_specific_config_update'")
        return config

    def set_step_epoch(self, step, epoch):
        self.total_step = step * epoch
        self.total_epoch = epoch

    """ basic methods """

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
