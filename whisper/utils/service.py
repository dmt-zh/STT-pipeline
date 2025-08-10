import gc
import logging
import numpy as np
import os
import re
import sys
import torch
import yaml

from collections.abc import Mapping, Sequence
from torch import Tensor
from typing import TypeAlias, Union

##############################################################################################

PipelineArg: TypeAlias = Union[
    bool,
    int,
    float,
    str,
    Sequence[int | float],
]
PipelineArgs: TypeAlias = Mapping[str, PipelineArg]

WhisperFeatures: TypeAlias = Sequence[Mapping[str, Union[Sequence[int], Tensor]]]

##############################################################################################

def setup_environment(n_threads: int = 1) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    logging.info('✔ Cuda cache cleaned and carbage is collected!')
    torch.set_num_threads(n_threads)
    logging.info(f'✔ Torch threads are set to "{n_threads}"')

##############################################################################################

def setup_random_seed(seed: int = 44) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f'🔑 Seed is set to "{seed}" for numpy and torch.')

##############################################################################################

def load_config() -> Mapping[str, PipelineArgs]:
    config_name = 'config.yml' if len(sys.argv) == 1 else sys.argv[1]
    with open(config_name) as config_yml:
        config = yaml.safe_load(config_yml)
    logging.info('✔ Config is successfully loaded!')
    return config

##############################################################################################

def slice_converter(patt: str) -> slice:
    if not patt:
        return False
    patt = str(patt)
    if re.fullmatch(r':-?\d+', patt):
        return slice(None, int(patt.replace(':', '')), None)
    if re.fullmatch(r'-?\d+:', patt):
        return slice(int(patt.replace(':', '')), None, None)
    if re.fullmatch(r'::\d+', patt):
        return slice(None, None, int(patt.replace(':', '')))
    logging.info(f'❌ ERROR. Invalid slice pattern passed. Valid options: ":±int", "±int:" or "::int"')
    os._exit(os.EX_OSFILE)

##############################################################################################
