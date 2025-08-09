#!.venv/bin/python3

import logging
from clearml import Task, Logger as CMLogger

from utils.preprocessing import WhisperDataset
from utils.service import (
    load_config,
    setup_environment,
    setup_random_seed,
)

##############################################################################################

logging_config = {
    'level': logging.INFO,
    'format': '%(asctime)s | %(message)s'
}
logging.basicConfig(**logging_config, datefmt='%Y-%m-%d %H:%M:%S')

##############################################################################################

config = load_config()
data_config = config.get('data')
train_config = config.get('training_args')

##############################################################################################

setup_environment(n_threads=config.get('setup', {}).get('torch_threads', 1))
setup_random_seed(seed=config.get('setup', {}).get('seed', 44))

##############################################################################################

training_data = WhisperDataset(config=data_config)
training_data.fetch()

clearml_task = Task.init(
    project_name=full_config.get('clearml').get('project_name'),
    task_name=full_config.get('clearml').get('experiment_name'),
    tags=full_config.get('clearml').get('tags'),
    auto_connect_frameworks={'pytorch': False}
)

##############################################################################################

clearml_task.connect_configuration(
    configuration=train_config,
    name='config.yml',
    description='Finetuning config'
)
clearml_task.close()
