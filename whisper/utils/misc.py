import logging
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from os import environ
from pathlib import Path
from random import choices
from typing import Any

from clearml import Task
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from torch import Tensor, nn
from transformers import (
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from utils.service import PipelineArgs, WhisperFeatures

##############################################################################################

def make_inputs_require_grad(module: nn.Conv1d, input_tensors: tuple[Tensor], output_tensor: Tensor):
    output_tensor.requires_grad_(True)

##############################################################################################

def fetch_model_and_processor(config: Mapping[str, PipelineArgs]) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    model_name = f'{config.get("model_name")}-{config.get("model_size")}'
    model_download_path = Path(config.get('cache_dir', '~/.cache/huggingface/hub')).resolve()
    model_download_path.mkdir(exist_ok=True, parents=True)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=model_download_path,
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        language=config.get('language'),
        task=config.get('task'),
        cache_dir=model_download_path,
        token=environ.get('HF_TOKEN'),
        trust_remote_code=True,
    )
    processor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    logging.info('Initialized processor with WhisperProcessor.')
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=model_download_path,
        device_map='auto',
    )
    prepared_model = prepare_model_for_kbit_training(model)
    prepared_model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    logging.info('Initialized and prepared Whisper model for training.')
    return prepared_model, processor

##############################################################################################

def init_lora_config(config: Mapping[str, PipelineArgs]) -> LoraConfig:
    """Инициализация Lora конфига для тренировки.
    Параметры Lora: https://huggingface.co/docs/peft/package_reference/lora
    """

    lora_config = LoraConfig(
        lora_alpha=config.get('alpha', 16),
        lora_dropout=config.get('dropout', 0.1),
        r=config.get('r', 16),
        bias=config.get('bias', 'none'),
        target_modules=config.get('target_modules', 'all-linear'),
    )
    logging.info('Initialized lora config with LoraConfig.')
    return lora_config

##############################################################################################

def init_train_config(config: Mapping[str, PipelineArgs]) -> Seq2SeqTrainingArguments:
    """
    https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    """
    prefix = config.get('model').get('prefix', 'whisper_tune')
    train_args = config.get('training_args')
    output_dir = Path(train_args.get('output_dir')).resolve()
    curr_dir = Path(__file__).resolve().parent
    default_log_dir = curr_dir.joinpath('logs')
    loggin_dir = Path(train_args.get('logging_dir', default_log_dir)).resolve()
    save_dir = output_dir.joinpath(prefix)

    train_config = Seq2SeqTrainingArguments(
        output_dir=str(save_dir),
        eval_strategy=train_args.get('eval_strategy', 'no'),
        eval_steps=train_args.get('eval_steps', None),
        eval_on_start=train_args.get('eval_on_start', False),
        per_device_train_batch_size=train_args.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=train_args.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=train_args.get('gradient_accumulation_steps', 1),
        learning_rate=train_args.get('learning_rate', 5e-5),
        weight_decay=train_args.get('weight_decay', 0),
        max_grad_norm=train_args.get('max_grad_norm', 1.0),
        num_train_epochs=train_args.get('num_train_epochs', 3),
        max_steps=train_args.get('max_steps', -1),
        lr_scheduler_type=train_args.get('lr_scheduler_type', 'linear'),
        logging_dir=str(loggin_dir),
        logging_strategy=train_args.get('logging_strategy', 'steps'),
        logging_steps=train_args.get('logging_steps', 500),
        logging_first_step=train_args.get('logging_first_step', False),
        save_strategy=train_args.get('save_strategy', 'steps'),
        save_steps=train_args.get('save_steps', 500),
        save_total_limit=train_args.get('save_total_limit', None),
        fp16=train_args.get('fp16', False),
        bf16=train_args.get('bf16', False),
        remove_unused_columns=train_args.get('remove_unused_columns', True),
        load_best_model_at_end=train_args.get('load_best_model_at_end', False),
        greater_is_better=train_args.get('greater_is_better', None),
        metric_for_best_model=train_args.get('metric_for_best_model', 'loss'),
        optim=train_args.get('optim', 'adamw_torch'),
        group_by_length=train_args.get('group_by_length', False),
        label_names=train_args.get('label_names', None),
        report_to=train_args.get('report_to', 'none'),
        seed=config.get('setup', {}).get('seed', 44),
    )
    logging.info('Initialized training config with Seq2SeqTrainingArguments.')
    return train_config

##############################################################################################

@dataclass
class SpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: WhisperFeatures) -> Mapping[str, Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')
        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels
        return batch

##############################################################################################

class ClearMLCallback(TrainerCallback):
    def __init__(self, clearml_task: Task, processor: WhisperProcessor, trainer: Trainer, k: int) -> None:
        self._clearml_task = clearml_task
        self._processor = processor
        self._trainer = trainer
        self._k = k

    ##########################################################################################

    def on_log(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PeftModel | None = None,
        processing_class: WhisperFeatureExtractor | None = None,
        logs: Mapping[str, int | float] | None = None,
        **kwargs,
    ):
        if 'eval_loss' in logs:
            self._clearml_task.get_logger().report_scalar(
                title='Loss',
                series='eval',
                value=logs.get('eval_loss'),
                iteration=state.global_step,
            )
        if 'train_runtime' in logs:
            if state.best_metric:
                self._clearml_task.get_logger().report_single_value(
                    name='Best metric',
                    value=round(state.best_metric, 5),
                )
                self._clearml_task.get_logger().report_single_value(
                    name='Checkpoint',
                    value=state.best_global_step,
                )

            self._clearml_task.get_logger().report_single_value(
                name='Epochs',
                value=round(logs.get('epoch'), 2),
            )
            self._clearml_task.get_logger().report_single_value(
                name='Runtime',
                value=logs.get('train_runtime'),
            )
            self._clearml_task.get_logger().report_single_value(
                name='Samples/sec',
                value=logs.get('train_samples_per_second'),
            )
            self._clearml_task.get_logger().report_single_value(
                name='Steps/sec',
                value=logs.get('train_steps_per_second'),
            )
        if 'learning_rate' in logs:
            self._clearml_task.get_logger().report_scalar(
                title='Loss',
                series='train',
                value=logs.get('loss'),
                iteration=state.global_step,
            )
            self._clearml_task.get_logger().report_scalar(
                title='train: learning rate',
                value=logs.get('learning_rate'),
                series='learning rate',
                iteration=state.global_step,
            )
            self._clearml_task.get_logger().report_scalar(
                title='train: norm gradients',
                value=logs.get('grad_norm'),
                series='gradients norm',
                iteration=state.global_step,
            )

        sample_ids = choices(range(len(self._trainer.eval_dataset)), k=self._k)
        samples = self._trainer.eval_dataset.select(sample_ids)
        samples_audio_path = samples['path']
        samples_sentences = samples['sentence']
        predicted_labels = self._trainer.predict(samples).label_ids
        decoded_samples = self._processor.batch_decode(predicted_labels, skip_special_tokens=True)

        for idx, (input_str, output_str, audi_path) in enumerate(zip(samples_sentences, decoded_samples, samples_audio_path, strict=False), 1):
            self._clearml_task.get_logger().report_media(
                title='eval samples',
                series=f'text-{idx}',
                iteration=state.global_step,
                stream=StringIO(f'{input_str} | original text\n{output_str} | predicted text'),
                file_extension='.txt',
            )
            self._clearml_task.get_logger().report_media(
                title='eval samples',
                series=f'audio-{idx}',
                iteration=state.global_step,
                local_path=audi_path,
                delete_after_upload=False,
            )

##############################################################################################
