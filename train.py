import hydra
import os
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mlp.base import common_functions as cmf
from mlp.launch import prepare  # NOQA
from mlp.utils import timer
# import torch
# torch.autograd.set_detect_anomaly(True)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def _train(cfg: DictConfig):
    cmf.create_save_dirs(cfg.path.working_dir)
    return train(cfg)


def train(config: DictConfig):
    """ Training the network """

    # create loggers
    it_logger = cmf.create_logger(config, "ITER", "train.log")
    eval_logger = cmf.create_logger(config, "EPOCH", "scores.log")

    # set random seed
    cmf.seed_everything(config.seed)
    
    """ Prepare data loader and model"""
    datamodule = config.data
    D = cmf.get_dataset(datamodule.dataname)
    # Since babel does not have a test set, use the validation set as the test set
    dsets, L = cmf.get_loader(
        D, split=["train", "test"], data_config=datamodule)

    # The following code is written for fine-tuning the language model using LoRA technology
    if config.resume:
        ckpt_dir = config.resume
        assert len(ckpt_dir) > 0
        ckpt_dir = os.path.join(str(config.path.code_dir), ckpt_dir)
        OmegaConf.update(config, "model.text.ckpt_dir", ckpt_dir, force_add=True)

    net, init_step = cmf.factory_model(config, dsets["train"], it_logger)

    it_logger.info('\nTotal params: %.2fM' % (sum(p.numel() for p in net.parameters() if p.requires_grad) / 1000000.0))

    # Prepare tensorboard
    net.create_tensorboard_summary(config.path.working_dir)

    """ Run training network """
    eval_every = config.machine["every_eval"]  # epoch
    eval_after = config.machine["after_eval"]  # epoch
    print_every = config.machine["print_every"]  # iteration
    num_step = config.machine["num_epochs"]  # epoch
    apply_cl_after = config.machine["curriculum_learning_at"]

    vis_every = config.machine["vis_every"]  # epoch
    if vis_every > 0:
        # Half of the training set data, half of the validation set data
        nsamples = config.machine["vis_nsamples"]
        vis_data = dsets["train"].get_samples(int(nsamples / 2))
        vis_data.extend(dsets["test"].get_samples(int(nsamples / 2)))
        vis_data = dsets["train"].collate_fn(vis_data)
        vis_inp, vis_gt = net.prepare_batch(vis_data)
        net.visualize(vis_inp, vis_gt, "epoch{:0>3d}".format(0))

    # We evaluate initialized model
    # cmf.test(config, L["test"], net, 0, eval_logger, mode="Valid")
    ii = 1
    net.train_mode()  # set network as train mode
    net.reset_status()  # initialize status
    tm = timer.Timer()  # tm: timer
    print("=====> # of iteration per one epoch: {}".format(len(L["train"])))
    # Set the total training step
    net.set_step_epoch(len(L["train"]), num_step)
    for epoch in range(init_step, num_step + 1):
        # curriculum learning
        if (apply_cl_after > 0) and (epoch == apply_cl_after):
            net.apply_curriculum_learning()

        for batch in tqdm(L["train"]):

            # Forward and update the network
            data_load_duration = tm.get_duration()
            tm.reset()
            net_inps, gts = net.prepare_batch(batch)
            outputs = net.forward_update(net_inps, gts)
            run_duration = tm.get_duration()

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs["net_output"], gts)

            # print learning status
            if (print_every > 0) and (ii % print_every == 0):
                net.print_status()
                lr = net.get_lr()
                txt = "fetching for {:.3f}s, optimizing for {:.3f}s, lr = {:.5f}"
                it_logger.info(txt.format(
                    data_load_duration, run_duration, lr))

            # for debugging
            if config.debug and (ii > 2):
                cmf.test(config, L["test"], net, 0, eval_logger, mode="Test")
                break

            tm.reset()
            ii = ii + 1
            # iteration done

        # visualize network learning status
        if (vis_every > 0) and (epoch % vis_every == 0):
            net.visualize(vis_inp, vis_gt, "epoch{:0>3d}".format(epoch))

        # validate current model
        if (epoch > eval_after) and (epoch % eval_every == 0):
            # print training losses
            # net.save_results("latest", mode="Train")
            net.print_counters_info(eval_logger, epoch, mode="Train")

            cmf.test(config, L["test"], net, epoch, eval_logger, mode="Test")

            net.train_mode()  # set network as train mode
            net.reset_status()  # initialize status

    print("=====> Training complete, safe and sound")


if __name__ == "__main__":
    # train network
    _train()
