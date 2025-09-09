#!.venv/bin/python3

import logging
from collections.abc import Mapping
from os import environ
from pathlib import Path

from clearml import Task
from clearml.utilities.proxy_object import flatten_dictionary
from dotenv import load_dotenv
from peft import get_peft_model
from transformers import Seq2SeqTrainer

from utils.dataset import WhisperDataset
from utils.inference import AdapterEvaluator
from utils.misc import (
    ClearMLCallback,
    SpeechSeq2SeqWithPadding,
    fetch_model_and_processor,
    init_lora_config,
    init_train_config,
)
from utils.service import (
    PipelineArgs,
    get_trainable_params,
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

def local_train(config: Mapping[str, PipelineArgs]) -> None:
    merge_and_evaluate = AdapterEvaluator(config=config.get('model'))
    whisper_model, whisper_processor = fetch_model_and_processor(config=config.get('model'))
    whisper_dataset = WhisperDataset(
        config=config.get('data'),
        processor=whisper_processor,
        seed=config.get('setup', {}).get('seed', 44),
    )
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
        peft_model.config.use_cache = False
        logging.info('✔ Started training!')
        whisper_trainer.train()
        best_model_checkpoint = whisper_trainer.state.best_model_checkpoint
        if best_model_checkpoint:
            merge_and_evaluate(
                dataset=train_dataset['test'],
                best_chkpt_path=best_model_checkpoint,
                adapters_path=None,
                iter_chkpt=False,
            )
        else:
            logging.info('WARNING. No "best_model_checkpoint" was gained. Skipping merging and evaluation.')
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
        description='The configuration of finetuning task'
    )
    merge_and_evaluate = AdapterEvaluator(config=config.get('model'), clearml_task=clearml_task)
    whisper_model, whisper_processor = fetch_model_and_processor(config=config.get('model'))
    whisper_dataset = WhisperDataset(
        config=config.get('data'),
        processor=whisper_processor,
        seed=config.get('setup', {}).get('seed', 44),
    )
    train_dataset = whisper_dataset.fetch()
    lora_config = init_lora_config(config=config.get('lora_params'))
    train_config = init_train_config(config=config)
    peft_model = get_peft_model(whisper_model, lora_config)
    
    lora_params = {k: sorted(v) if isinstance(v, set) else v for k, v in lora_config.to_dict().items()}
    train_params = {k: sorted(v) if isinstance(v, set) else v for k, v in train_config.to_dict().items()}

    clearml_task._arguments.copy_from_dict(
        flatten_dictionary(lora_params),
        prefix='LoraConfig',
    )
    clearml_task._arguments.copy_from_dict(
        flatten_dictionary(train_params),
        prefix='TrainingArguments',
    )
    clearml_task._arguments.copy_from_dict(
        get_trainable_params(peft_model),
        prefix='TrainableParams',
    )
    clearml_task.connect_configuration(
        configuration=peft_model.config.to_dict(),
        name='WhisperModel',
        description='The configuration of Whisper model',
    )
    clearml_task.upload_artifact(
        name='Model architecture',
        artifact_object=repr(peft_model),
        sort_keys=False,
    )
    whisper_trainer = Seq2SeqTrainer(
        args=train_config,
        model=peft_model,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        data_collator=SpeechSeq2SeqWithPadding(processor=whisper_processor),
        processing_class=whisper_processor.feature_extractor,
    )
    clearml_callback = ClearMLCallback(
        clearml_task=clearml_task,
        processor=whisper_processor,
        trainer=whisper_trainer,
        k=config.get('data').get('debug_samples', 1),
    )
    whisper_trainer.add_callback(clearml_callback)
    if config.get('training_args').get('do_train', False):
        peft_model.config.use_cache = False
        whisper_trainer.train()
        best_model_checkpoint = whisper_trainer.state.best_model_checkpoint
        best_model_path = Path(best_model_checkpoint).resolve()
        logging.info(f'Evaluating best checkpoint "{best_model_path.name}"')
        if best_model_checkpoint:
            merge_and_evaluate(
                dataset=train_dataset['test'],
                best_chkpt_path=best_model_checkpoint,
                iter_step=whisper_trainer.state.global_step + 1,
                adapters_path=None,
                iter_chkpt=False,
            )
            logging.info(f'Uploading best adapter "{best_model_path.name}" to S3.')
            clearml_task.upload_artifact(
                name=best_model_path.name,
                artifact_object=best_model_checkpoint,
                delete_after_upload=False,
            )
            if whisper_trainer.state.best_global_step != whisper_trainer.state.global_step:
                last_chkpt_name = f'checkpoint-{whisper_trainer.state.global_step}'
                last_model_path = best_model_path.parent.joinpath(last_chkpt_name)
                logging.info(f'Evaluating last checkpoint "{last_chkpt_name}"')
                merge_and_evaluate(
                    dataset=train_dataset['test'],
                    best_chkpt_path=str(last_model_path),
                    iter_step=whisper_trainer.state.global_step + 2,
                    last_chkpt=True,
                )
                logging.info(f'Uploading last checkpoint "{last_chkpt_name}" to S3.')
                clearml_task.upload_artifact(
                    name=last_chkpt_name,
                    artifact_object=str(last_model_path),
                    delete_after_upload=False,
                )
        else:
            logging.info('WARNING. No "best_model_checkpoint" was gained. Skipping merging and evaluation.')
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
