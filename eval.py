import hydra
import os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from mlp.base import common_functions as cmf
from mlp.launch import prepare  # NOQA


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def eval(newcfg: DictConfig) -> None:
    # load model and options
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    ckpt_path = newcfg.last_ckpt_path
    split = newcfg.split

    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    config = OmegaConf.merge(prevcfg, newcfg)

    eval_logger = cmf.create_logger(config, "EPOCH", "eval.log")

    # set random seed
    cmf.seed_everything(config.seed)

    OmegaConf.update(config, "model.text.ckpt_dir",
                     os.path.dirname(ckpt_path), force_add=True)
    OmegaConf.update(config, "model.eval_checked", True, force_add=True)
    # for visualise "false-negative moments"
    # OmegaConf.update(config, "model.training_checked", True, force_add=True)
    OmegaConf.update(config, "model.save_pred", True, force_add=True)
    
    # prepare dataset
    D = cmf.get_dataset(config.data.dataname)
    dsets, L = cmf.get_loader(D, split=[split], data_config=config.data)        

    # Build network
    net, _ = cmf.factory_model(config, dsets[split], eval_logger)
    net.load_checkpoint(ckpt_path, True)

    # Evaluating networks
    cmf.test(config, L[split], net, -1, None, mode="Test")
    # random visualize validation set data
    nsamples = config.machine["vis_nsamples"]
    vis_data = dsets[split].get_samples(int(nsamples)*2)
    vis_data = dsets[split].collate_fn(vis_data)
    vis_inp, vis_gt = net.prepare_batch(vis_data)
    net.visualize(vis_inp, vis_gt, "Test-{:0>3d}".format(-1))


if __name__ == "__main__":
    eval()
