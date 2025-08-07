import os
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
from logging import Logger
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, Union

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

def _common_voice_restruct(sample: LazyRow) -> LazyRow:
    sample['path'] = sample['audio']['path']
    return sample

##############################################################################################

def _fleurs_restruct(sample: LazyRow) -> LazyRow:
    sample_dir = Path(sample['path']).resolve().parent
    audio_path = sample_dir.joinpath(sample['audio']['path'])
    sample['path'] = str(audio_path)
    return sample

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
    def __init__(
        self,
        config: Mapping[str, DatasetArgs],
        logger: Logger,
    ) -> None:
        self._config = config
        self._logger = logger
        self._add_noise = self._is_required('noise', config)
        self._common_voice = self._is_required('common_voice', config)
        self._fleurs = self._is_required('fleurs', config)
        self._datasets = {
            'common_voice': 'Common Voice',
            'fleurs': 'Fleurs',
        }

    ##########################################################################################

    def _is_required(self, name: str, config: Mapping[str, DatasetArgs]) -> bool:
        is_required = config.get(name, {}).get('include', False)
        if is_required:
            self._logger.info(f'Initialized adding "{name}" to training dataset.')
        return is_required

    ##########################################################################################

    def _load_dataset_from_hf(self, config: Mapping[str, DatasetArgs]) -> ArrowDataset:
        download_dir = config.get('cache_dir', '~/.cache/huggingface/datasets')
        if not Path(download_dir).resolve().exists():
            self._logger.info(f'Loading "{self._datasets.get('name')}" dataset.')

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
        return cleaned_dataset

    ##########################################################################################

    def _load_and_process_common_voice(self) -> ArrowDataset:
        dataset_config = self._config.get('common_voice')
        raw_dataset = self._load_dataset_from_hf(dataset_config)
        self._logger.info(f'Processing "Common Voice" to restructure samples...')
        restructed_dataset = raw_dataset.map(
            _common_voice_restruct,
            num_proc=dataset_config.get('num_proc')
        )
        processed_dataset = restructed_dataset.cast_column('audio', Audio(sampling_rate=16000))
        self._logger.info(f'Restructured "Common Voice" and casted audio to "sampling_rate=16000".')
        return processed_dataset

    ##########################################################################################

    def _load_and_process_fleurs(self) -> ArrowDataset:
        dataset_config = self._config.get('fleurs')
        raw_dataset = self._load_dataset_from_hf(dataset_config)
        restructed_dataset = raw_dataset.map(
            _fleurs_restruct,
            num_proc=dataset_config.get('num_proc')
        )
        self._logger.info(f'Restructured "Fleurs" dataset.')
        return restructed_dataset

    ##########################################################################################

    def _load_and_process_custom(self) -> None:
        dataset_config = self._config.get('custom')
        data_dir = Path(dataset_config.get('path')).resolve()
        if not data_dir.exists():
            raise ValueError(f'Provided directory "{str(data_dir)}" does not exist!')
        audio_dir = next(data_dir.glob('audio'))
        text_dir = next(data_dir.glob('sentences'))

        pass

    ##########################################################################################

    def _apply_noise(self) -> None:
        pass

    ##########################################################################################

    def fetch(self) -> None:
        datasets_to_concat = []
        if self._common_voice:
            common_voise_dataset = self._load_and_process_common_voice()
            self._logger.info(f'"Common Voice" dataset size is: {len(common_voise_dataset)} samples')
            datasets_to_concat.append(common_voise_dataset)
        if self._fleurs:
            fleurs_dataset = self._load_and_process_fleurs()
            self._logger.info(f'"Fleurs" dataset size is: {len(fleurs_dataset)} samples')
            datasets_to_concat.append(fleurs_dataset)

##############################################################################################
