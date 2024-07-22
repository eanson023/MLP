import logging
import time
import numpy as np
import os
import random
import shutil
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from mlp.data import babel, humanml3d
from mlp.utils import io_utils


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def get_dataset(dataset):
    if dataset == "babel":
        D = eval("babel")
    elif dataset == "humanml3d":
        D = eval("humanml3d")
    else:
        raise NotImplementedError(
            "Not supported dataset type ({})".format(dataset))
    return D


def get_loader(D, data_config, split=[]):
    assert len(split) > 0
    return D.create_loaders(split, data_config)


def factory_model(config, dset=None, logger=None):
    net = instantiate(config.model, logger=logger,
                      working_dir=config.path.working_dir,
                      use_gpu=config.machine['use_gpu'],
                      # Avoid recursive early loading of encoders
                      _recursive_=False)
    if dset is not None:
        net.bring_dataset_info(dset)

    # load checkpoint
    epoch = 1
    if config.resume:
        ckpt_dir = config.resume
        assert len(ckpt_dir) > 0
        ckpt_path = os.path.join(str(config.path.code_dir), ckpt_dir, 'ckpt.pkl')
        epoch, it = net.load_checkpoint(ckpt_path, True)
        net.it = it
        print(f'Continue training from {config.resume}, epoch: {epoch}, iteration: {it}')

    # ship network to use gpu
    if config.machine["use_gpu"]:
        net.gpu_mode()
    if logger is not None:
        logger.info(net)
        time.sleep(0.5)
    return net, epoch


def create_save_dirs(root_path):
    """ Create neccessary directories for training and evaluating models
    """
    # create directory for checkpoints
    io_utils.check_and_create_dir(os.path.join(
        root_path, "checkpoints"))
    # create directory for results
    io_utils.check_and_create_dir(os.path.join(root_path, "status"))
    io_utils.check_and_create_dir(os.path.join(
        root_path, "qualitative"))


def create_logger(config, logger_name, log_path):
    """ Get logger """
    logger_path = os.path.join(config.path.working_dir, log_path)
    logger = io_utils.get_logger(
        logger_name, log_file_path=logger_path,
        print_lev=getattr(logging, config["logging"]["print_level"]),
        write_lev=getattr(logging, config["logging"]["write_level"]))
    return logger


def test(config, loader, net, epoch, eval_logger=None, mode="Test", count_loss=False):
    """ evaluate the network """
    with torch.no_grad():
        net.eval_mode()  # set network as evaluation mode
        net.reset_status()  # reset status
        net.reset_counters()

        # Testing network
        ii = 1
        for batch in tqdm(loader, desc="{}".format(mode)):
            # forward the network
            net_inps, gts = net.prepare_batch(batch)
            if count_loss:
                # compute loss and forward
                outputs = net.compute_loss(net_inps, gts, mode)
            else:
                outputs = net.forward_only(net_inps, mode)  # only forward

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs["net_output"], gts)

            ii += 1
            if config.debug and (ii > 3):
                break
            # end for batch in loader
        
        # save result
        prefix_name = "latest"
        # prefix_name = f"epoch{epoch+1:0>3d}"
        net.save_results(prefix_name, mode=mode)
        if epoch > 0:
            ckpt_dir = os.path.join(config.path.working_dir, "checkpoints", prefix_name)
            io_utils.check_and_create_dir(ckpt_dir)
            net.save_checkpoint(os.path.join(ckpt_dir, "ckpt.pkl"), epoch + 1, save_crit=True)
        if net.renew_best_score() and epoch > 0:
            prefix_name = "model_best"
            net.save_results(prefix_name, mode=mode)
            tgt_ckpt_dir = os.path.join(config.path.working_dir, "checkpoints", prefix_name)
            if os.path.exists(tgt_ckpt_dir):
                shutil.rmtree(tgt_ckpt_dir)
            # Copy the current weight to the model_best folder
            shutil.copytree(ckpt_dir, tgt_ckpt_dir)
        
        net.print_counters_info(eval_logger, epoch, mode=mode)


def extract_output(config, loader, net, save_dir):
    """ miscs """

    with torch.no_grad():
        net.eval_mode()  # set network as evaluation mode
        net.reset_status()  # reset status
        net.reset_counters()

        # Testing network
        ii = 1
        for batch in tqdm(loader, desc="extract_output"):
            # forward the network
            net_inps, gts = net.prepare_batch(batch)
            outputs = net.extract_output(
                net_inps, gts, save_dir)  # only forward

            ii += 1
            if config["misc"]["debug"] and (ii > 3):
                break
            # end for batch in loader


""" Methods for debugging """


def one_step_forward(L, net, logger):
    # fetch the batch
    batch = next(iter(L))

    # forward and update the network
    outputs = net.forward_update(batch)

    # accumulate the number of correct answers
    net.compute_status(outputs, batch["gt"])

    # print learning status
    net.print_status(1, logger)
