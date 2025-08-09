import logging
import os
from copy import deepcopy
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    interleave_datasets, 
    IterableDataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.formatting.formatting import LazyRow
from datasets.utils.logging import disable_progress_bar
from collections.abc import Mapping, Sequence
from numpy.random import RandomState
from pathlib import Path
from typing import Any, TypeAlias, Union
from utils.service import slice_converter

##############################################################################################

disable_progress_bar()

##############################################################################################

def transform_and_tokenize_batch(feature_extractor, tokenizer, batch):
    audio = batch['audio']
    batch['input_features'] = feature_extractor(
        audio['array'], 
        sampling_rate=audio['sampling_rate'],
        do_normalize=True
    ).input_features[0]
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    return batch

##############################################################################################

def _generate_random_sequence(length: int, seed: int) -> Sequence[int]:
    random_generator = RandomState(seed)
    return random_generator.permutation(length)

##############################################################################################

def _fleurs_restruct(sample: LazyRow) -> LazyRow:
    sample_dir = Path(sample['path']).resolve().parent
    audio_path = sample_dir.joinpath(sample['audio']['path'])
    sample['audio']['path'] = str(audio_path)
    sample['sentence'] = sample['raw_transcription']
    del sample['path']
    del sample['raw_transcription']
    return sample

##############################################################################################

def _get_text_samples(text_files: Sequence[str | Path]) -> Sequence[str]:
    text_buffer = []
    for text_file in text_files:
        with open(text_file, encoding='utf8') as fin:
            text_lines = tuple(map(str.strip, fin.readlines()))
            if text_lines:
                for idx, text_line in enumerate(text_lines):
                    if not text_line:
                        logging.info(f'📢 WARNING. Text file "{text_file.name}" contains empty line at index "{idx}".')
                    text_buffer.append(text_line)
            else:
                logging.info(f'📢 WARNING. Text file "{text_file.name}" is empty.')
    return tuple(text_buffer)

##############################################################################################

DatasetArg: TypeAlias = Union[
    bool,
    int,
    float,
    str,
    Sequence[int | float],
]
DatasetArgs: TypeAlias = Mapping[str, DatasetArg]

##############################################################################################

class WhisperDataset:
    def __init__(self, config: Mapping[str, DatasetArgs]) -> None:
        self._config = config
        self._add_noise = self._is_required('noise', config)
        self._common_voice = self._is_required('common_voice', config)
        self._fleurs = self._is_required('fleurs', config)
        self._custom = self._is_required('custom', config)

    ##########################################################################################

    def _is_required(self, name: str, config: Mapping[str, DatasetArgs]) -> bool:
        is_required = config.get(name, {}).get('include', False)
        if is_required:
            logging.info(f'≫ Initialized adding "{name}" to training dataset.')
        return is_required

    ##########################################################################################

    def _load_dataset_from_hf(self, config: Mapping[str, DatasetArgs]) -> ArrowDataset:
        download_dir = config.get('cache_dir', '~/.cache/huggingface/datasets')
        if not Path(download_dir).resolve().exists():
            logging.info(f'Loading "{config.get("path")}" dataset.')

        raw_dataset = load_dataset(
            path=config.get('path'),
            name=config.get('lang'),
            split=config.get('split'),
            cache_dir=download_dir,
            token='', ##FIXME: load from .env file
            num_proc=config.get('num_proc'),
            trust_remote_code=True,
        )
        columns_to_remove = [
            col for col in raw_dataset.column_names if col not in ('audio', 'sentence', 'raw_transcription', 'path')
        ]
        cleaned_dataset = raw_dataset.remove_columns(columns_to_remove)
        slice_patt = slice_converter(config.get('slice'))
        if slice_patt:
            cleaned_dataset = Dataset.from_dict(cleaned_dataset[slice_patt])

        if config.get('shuffle', False):
            cleaned_dataset = cleaned_dataset.shuffle(
                seed=config.get('seed', 123),
                keep_in_memory=True,
            )
        return cleaned_dataset

    ##########################################################################################

    def _load_and_process_common_voice(self) -> ArrowDataset:
        dataset_config = self._config.get('common_voice')
        raw_dataset = self._load_dataset_from_hf(dataset_config)
        cleaned_dataset = raw_dataset.remove_columns('path')
        processed_dataset = cleaned_dataset.cast_column('audio', Audio(sampling_rate=16000))
        logging.info(f'Restructured "Common Voice" and casted audio to "sampling_rate=16000".')
        return processed_dataset

    ##########################################################################################

    def _load_and_process_fleurs(self) -> None:
        dataset_config = self._config.get('fleurs')
        raw_dataset = self._load_dataset_from_hf(dataset_config)
        restructed_dataset = raw_dataset.map(
            _fleurs_restruct,
            num_proc=dataset_config.get('num_proc')
        )
        processed_dataset = restructed_dataset.cast_column('audio', Audio(sampling_rate=16000))
        logging.info(f'Restructured "Fleurs" and casted audio to "sampling_rate=16000".')
        return processed_dataset

    ##########################################################################################

    def fetch(self) -> None:
        datasets_to_concat = []
        if self._common_voice:
            common_voise_dataset = self._load_and_process_common_voice()
            logging.info(f'"Common Voice" dataset size is: {len(common_voise_dataset)} samples.')
            datasets_to_concat.append(common_voise_dataset)
        if self._fleurs:
            fleurs_dataset = self._load_and_process_fleurs()
            logging.info(f'"Fleurs" dataset size is: {len(fleurs_dataset)} samples.')
            datasets_to_concat.append(fleurs_dataset)

        combined_dataset = concatenate_datasets(datasets_to_concat)
        logging.info(f'≫ Combined dataset for tuning Whisper has size "{len(combined_dataset)}" samples.')

##############################################################################################
