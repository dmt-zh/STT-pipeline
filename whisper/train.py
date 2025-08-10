#!.venv/bin/python3

import logging
from dotenv import load_dotenv
from clearml import Task, Logger as CMLogger
from collections.abc import Mapping
from os import environ
from pathlib import Path
from peft import get_peft_model
from transformers import Seq2SeqTrainer
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import WhisperDataset
from utils.misc import (
    fetch_model_and_processor,
    init_lora_config,
    init_train_config,
    SpeechSeq2SeqWithPadding,
)
from utils.service import (
    load_config,
    setup_environment,
    setup_random_seed,
    PipelineArgs
)

##############################################################################################

logging_config = {
    'level': logging.INFO,
    'format': '%(asctime)s | %(message)s'
}
logging.basicConfig(**logging_config, datefmt='%Y-%m-%d %H:%M:%S')

##############################################################################################

def local_train(config: Mapping[str, PipelineArgs]) -> None:
    whisper_model, whisper_processor = fetch_model_and_processor(config=config.get('model'))
    whisper_dataset = WhisperDataset(config=config.get('data'), processor=whisper_processor)
    train_dataset = whisper_dataset.fetch()
    lora_config = init_lora_config(config=config.get('lora_params'))
    train_config = init_train_config(config=config)
    peft_model = get_peft_model(whisper_model, lora_config)

    whisper_trainer = Seq2SeqTrainer(
        args=train_config,
        model=peft_model,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        data_collator=SpeechSeq2SeqWithPadding(processor=whisper_processor),
        processing_class=whisper_processor.feature_extractor,
    )
    if config.get('training_args').get('do_train', False):
        peft_model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        logging.info('✔ Started training!')
        whisper_trainer.train()
    else:
        logging.info('Running trainer is disabled.')

##############################################################################################

def clearml_train(config: Mapping[str, PipelineArgs]) -> None:

    clearml_task = Task.init(
        project_name=config.get('clearml').get('project_name'),
        task_name=config.get('clearml').get('experiment_name'),
        tags=config.get('clearml').get('tags'),
        output_uri=environ.get('CLEARML_URI'),
        auto_connect_frameworks={'pytorch': False, 'transformers': False}
    )
    clearml_task.connect_configuration(
        configuration=config,
        name='config.yml',
        description='Finetuning config'
    )
    whisper_model, whisper_processor = fetch_model_and_processor(config=config.get('model'))
    whisper_dataset = WhisperDataset(config=config.get('data'), processor=whisper_processor)
    train_dataset = whisper_dataset.fetch()
    lora_config = init_lora_config(config=config.get('lora_params'))
    train_config = init_train_config(config=config)
    model_to_train = get_peft_model(whisper_model, lora_config)

    writer = SummaryWriter()
    whisper_trainer = Seq2SeqTrainer(
        args=train_config,
        model=model_to_train,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        data_collator=SpeechSeq2SeqWithPadding(processor=whisper_processor),
        processing_class=whisper_processor.feature_extractor,
    )
    if config.get('training_args').get('do_train', False):
        model_to_train.config.use_cache = False # silence the warnings. Please re-enable for inference!
        whisper_trainer.train()
        clearml_task.close()
    else:
        logging.info('Running trainer is disabled.')

##############################################################################################

def main() -> None:
    load_dotenv()
    config = load_config()
    setup_environment(n_threads=config.get('setup', {}).get('torch_threads', 1))
    setup_random_seed(seed=config.get('setup', {}).get('seed', 44))
    if config.get('clearml').get('enable', False):
        clearml_train(config=config)
    else:
        local_train(config=config)

##############################################################################################

if __name__ == '__main__':
    main()
