import gc
import logging
import numpy as np
import torch
import sys
import yaml

from collections.abc import Mapping
from logging import Logger
from typing import Any

##############################################################################################

logging_config = {
    'level': logging.INFO,
    'format': '%(asctime)s | %(message)s (called: "%(funcName)s": from module: "%(pathname)s")'
}
logging.basicConfig(**logging_config, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

##############################################################################################

def setup_environment(logger: Logger, n_threads: int = 1) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    logger.info('\033[0;32m✔\033[0m Cuda cache cleaned and carbage is collected!')
    torch.set_num_threads(n_threads)
    logger.info(f'\033[0;32m✔\033[0m Torch threads are set to "{n_threads}"')

##############################################################################################

def setup_random_seed(logger: Logger, seed: int = 44) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f'🔑 Seed is set to "{seed}" for numpy and torch.')

##############################################################################################

def load_config(logger: Logger) -> Mapping[str, Any]:
    config_name = 'config.yml' if len(sys.argv) == 1 else sys.argv[1]
    with open(config_name) as config_yml:
        config = yaml.safe_load(config_yml)
    logger.info('\033[0;32m✔\033[0m Config is successfully loaded!')
    return config

##############################################################################################
