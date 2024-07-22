import hydra
import os
import warnings
from omegaconf import OmegaConf
from pathlib import Path


# Local paths
def code_path(path=""):
    code_dir = hydra.utils.get_original_cwd()
    code_dir = Path(code_dir)
    return code_dir / path


def working_path(path):
    return Path(os.getcwd()) / path


def get_last_checkpoint(path, ckpt_dir="model_best"):
    output_dir = Path(hydra.utils.to_absolute_path(path))
    last_ckpt_path = output_dir / "checkpoints" / ckpt_dir / "ckpt.pkl"
    return last_ckpt_path


OmegaConf.register_new_resolver("code_path", code_path)
OmegaConf.register_new_resolver("working_path", working_path)
OmegaConf.register_new_resolver("absolute_path", hydra.utils.to_absolute_path)
OmegaConf.register_new_resolver("get_last_checkpoint", get_last_checkpoint)

# Remove some warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

warnings.filterwarnings(
    "ignore", ".*pyprof will be removed by the end of June.*"
)

warnings.filterwarnings(
    "ignore", ".*pandas.Int64Index is deprecated.*"
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck*"
)

warnings.filterwarnings(
    "ignore", ".*Our suggested max number of worker in current system is*"
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "8"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
