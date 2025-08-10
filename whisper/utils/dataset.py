import logging
import os
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from datasets.formatting.formatting import LazyRow
from datasets.utils.logging import disable_progress_bar
from collections.abc import Mapping, Sequence
from numpy.random import RandomState
from pathlib import Path
from transformers import WhisperProcessor
from utils.service import slice_converter, PipelineArgs

##############################################################################################

disable_progress_bar()

##############################################################################################

def _generate_random_sequence(length: int, seed: int) -> Sequence[int]:
    random_generator = RandomState(seed)
    return random_generator.permutation(length)

##############################################################################################

def _common_voice_restruct(sample: LazyRow) -> LazyRow:
    sample['path'] = sample['audio']['path']
    return sample

##############################################################################################

def _fleurs_restruct(sample: LazyRow) -> LazyRow:
    sample_dir = Path(sample['path']).resolve().parent
    audio_path = sample_dir.joinpath(sample['audio']['path'])
    sample['path'] = str(audio_path)
    sample['audio']['path'] = str(audio_path)
    sample['sentence'] = sample['raw_transcription']
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

def _transform_and_tokenize_sample(sample: LazyRow, processor: WhisperProcessor) -> LazyRow:
    audio = sample['audio']
    sample['input_features'] = processor.feature_extractor(
        audio['array'], 
        sampling_rate=audio['sampling_rate'],
        do_normalize=True
    ).input_features[0]
    sample['labels'] = processor.tokenizer(sample['sentence']).input_ids
    return sample

##############################################################################################

class WhisperDataset:
    def __init__(self, config: Mapping[str, PipelineArgs], processor: WhisperProcessor) -> None:
        self._config = config
        self._processor = processor
        self._add_noise = self._is_required('noise', config)
        self._common_voice = self._is_required('common_voice', config)
        self._fleurs = self._is_required('fleurs', config)
        self._custom = self._is_required('custom', config)
        self._columns = ('audio', 'sentence', 'raw_transcription', 'path')

    ##########################################################################################

    def _is_required(self, name: str, config: Mapping[str, PipelineArgs]) -> bool:
        is_required = config.get(name, {}).get('include', False)
        if is_required:
            logging.info(f'Initialized adding "{name}" to training dataset.')
        return is_required

    ##########################################################################################

    def _load_dataset_from_hf(self, config: Mapping[str, PipelineArgs], name: str) -> Dataset:
        download_dir = config.get('cache_dir', '~/.cache/huggingface/datasets')
        download_path = Path(download_dir).resolve()
        if not download_path.exists():
            logging.info(f'Loading "{name}" dataset.')

        raw_dataset = load_dataset(
            path=config.get('path'),
            name=config.get('lang'),
            split=config.get('split'),
            cache_dir=download_path,
            token=os.environ.get('HF_TOKEN'),
            num_proc=config.get('num_proc'),
            trust_remote_code=True,
        )
        columns_to_remove = [col for col in raw_dataset.column_names if col not in self._columns]
        cleaned_dataset = raw_dataset.remove_columns(columns_to_remove)
        return cleaned_dataset

    ##########################################################################################

    def _load_and_process_common_voice(self) -> Dataset:
        raw_dataset = self._load_dataset_from_hf(
            config=self._config.get('common_voice'),
            name='common_voice'
        )
        restructed_dataset = raw_dataset.map(
            _common_voice_restruct,
            num_proc=self._config.get('common_voice').get('num_proc', 1)
        )
        processed_dataset = restructed_dataset.cast_column('audio', Audio(sampling_rate=16000))
        logging.info(f'Restructured "Common Voice" and casted audio to "sampling_rate=16000".')
        return processed_dataset

    ##########################################################################################

    def _load_and_process_fleurs(self) -> Dataset:
        raw_dataset = self._load_dataset_from_hf(
            config=self._config.get('fleurs'),
            name='fleurs',
        )
        restructed_dataset = raw_dataset.map(
            _fleurs_restruct,
            num_proc=self._config.get('fleurs').get('num_proc', 1)
        )
        processed_dataset = restructed_dataset.cast_column('audio', Audio(sampling_rate=16000))
        logging.info(f'Restructured "Fleurs" and casted audio to "sampling_rate=16000".')
        return processed_dataset

    ##########################################################################################

    def _load_and_process_custom(self) -> Dataset:
        dataset_config = self._config.get('custom')
        data_dir = Path(dataset_config.get('path')).resolve()
        if not data_dir.exists():
            logging.info(f'❌ ERROR. Provided directory "{str(data_dir)}" does not exist!')
            os._exit(os.EX_OSFILE)

        audio_dir = next(data_dir.glob('audio'))
        text_dir = next(data_dir.glob('sentences'))
        audio_files = tuple(sorted(str(audio) for audio in audio_dir.iterdir()))
        text_files = tuple(sorted(text_dir.iterdir()))
        diff_size = abs(len(audio_files) - len(text_files))

        if diff_size:
            logging.info(f'❌ ERROR. Number of audio and text files differs by "{diff_size}" items')
            os._exit(os.EX_OSFILE)

        slice_patt = slice_converter(dataset_config.get('slice'))
        if slice_patt:
            audio_files = tuple(audio_files[slice_patt])
            text_files = tuple(text_files[slice_patt])

        if dataset_config.get('shuffle', False):
            indices = _generate_random_sequence(len(text_files), dataset_config.get('seed', 123))
            audio_files = tuple(audio_files[idx] for idx in indices)
            text_files = tuple(text_files[idx] for idx in indices)

        processed_dataset = Dataset \
            .from_dict(
                {
                    'path': audio_files,
                    'audio': audio_files, 
                    'sentence': _get_text_samples(text_files)
                }
            ) \
            .cast_column('audio', Audio(sampling_rate=16000))

        logging.info(f'Casted custom audio dataset "{data_dir.name}" to "sampling_rate=16000".')
        return processed_dataset

    ##########################################################################################

    def _load_and_concat(self) -> Dataset:
        datasets_to_concat = []
        if self._common_voice:
            common_voise_dataset = self._load_and_process_common_voice()
            logging.info(f'"Common Voice" dataset size is: {len(common_voise_dataset)} samples.')
            datasets_to_concat.append(common_voise_dataset)
        if self._fleurs:
            fleurs_dataset = self._load_and_process_fleurs()
            logging.info(f'"Fleurs" dataset size is: {len(fleurs_dataset)} samples.')
            datasets_to_concat.append(fleurs_dataset)
        if self._custom:
            custom_dataset = self._load_and_process_custom()
            logging.info(f'"Custom" dataset size is: {len(custom_dataset)} samples.')
            datasets_to_concat.append(custom_dataset)

        if len(datasets_to_concat) == 1:
            combined_dataset = datasets_to_concat[0]
        else:
            combined_dataset = concatenate_datasets(datasets_to_concat)
        logging.info(f'Combined dataset for tuning Whisper has size "{len(combined_dataset)}" samples.')
        return combined_dataset

    ##########################################################################################

    def fetch(self) -> Dataset:
        preprocessed_dataset = self._load_and_concat()
        splited_dataset = preprocessed_dataset.train_test_split(
            test_size=self._config.get('eval_fraction', 0.1),
            shuffle=True
        )
        train_size = splited_dataset['train'].num_rows
        test_size = splited_dataset['test'].num_rows
        logging.info(f'Splited dataset to [train/test] parts: [{train_size}/{test_size}]')
        logging.info('Preparing dataset for training...')
        training_dataset = splited_dataset.map(
            _transform_and_tokenize_sample,
            num_proc=self._config.get('num_proc', 1),
            fn_kwargs={'processor': self._processor},
        )
        logging.info('Extracted features and tokenized dataset with WhisperProcessor.')
        return training_dataset

##############################################################################################
