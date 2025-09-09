import logging
import math
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import jiwer.transforms as tr
import numpy as np
from clearml import Task
from ctranslate2 import StorageView
from ctranslate2.models import Whisper
from datasets import Dataset
from jiwer import cer, wer
from peft import PeftModel
from pydub import AudioSegment
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from utils.html import generate_html
from utils.service import PipelineArgs, setup_environment

##############################################################################################

def _calculate_sent_wer_cer(hypothesis: str, reference: str):
    base_transforms = [
        tr.Strip(),
        tr.RemoveEmptyStrings(),
        tr.ToLowerCase(),
        tr.SubstituteRegexes({r'[。！？；： ，、《》〈〉『』 （）［］]': ' '}),
        tr.RemovePunctuation(),
        tr.RemoveMultipleSpaces(),
    ]
    wer_transforms = deepcopy(base_transforms) + [tr.ReduceToListOfListOfWords()]
    cer_transforms = deepcopy(base_transforms) + [tr.ReduceToListOfListOfChars()]
    wer_metric = wer(
        reference,
        hypothesis, 
        reference_transform=tr.Compose(wer_transforms), 
        hypothesis_transform=tr.Compose(wer_transforms),
    )
    cer_metric = cer(
        reference,
        hypothesis,
        reference_transform=tr.Compose(cer_transforms), 
        hypothesis_transform=tr.Compose(cer_transforms),
    )
    return wer_metric, cer_metric

##############################################################################################

def _chunks(audio: AudioSegment, n: int) -> AudioSegment:
    for i in range(0, math.ceil(audio.duration_seconds * 1000), n):
        yield audio[i:i + n]

##############################################################################################

def _recognize_audio_file(file_path: str, lang: str, model: Whisper, processor: WhisperProcessor) -> str:
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(16000)
    FRAME_SEC = 30
    transcription = ''
    for chunk in _chunks(audio, int(FRAME_SEC * 1000)):
        audio_arr = np.array(chunk.get_array_of_samples()).astype(np.float32)
        audio_arr = audio_arr / (1 << 8*2 - 1)
        inputs = processor(
            audio_arr, 
            return_tensors='np', 
            sampling_rate=audio.frame_rate,
            do_normalize=True,
        )
        features = StorageView.from_array(inputs.input_features)
        prompt = processor.tokenizer.convert_tokens_to_ids(
            [
                '<|startoftranscript|>',
                f'<|{lang}|>',
                '<|transcribe|>',
                '<|notimestamps|>',
            ]
        )
        results = model.generate(
            features,
            [prompt],
            beam_size=5,
            num_hypotheses=1,
            return_no_speech_prob=False,
        )
        tokens = results[0].sequences_ids[0]
        transcription += processor.decode(tokens, skip_special_tokens=True)
    return transcription.strip()

#################################################################################################

class AdapterEvaluator:
    def __init__(
        self,
        config: Mapping[str, PipelineArgs],
        clearml_task: Task | None = None,
    ) -> None:
        self._config = config
        self._clearml_task = clearml_task
        self._base_model_name = None
        self._base_model_cache = None
        self._chkpt = None
        self._feature_size = 128 if config.get('model_size').startswith('large') else 80

    ##########################################################################################

    def _merge_and_save(self, best_chkpt_path: str) -> Path:
        best_chkpt_path = Path(best_chkpt_path).resolve()
        self._chkpt = best_chkpt_path.name.split('-')[-1]
        merged_model_path = Path(self._config.get('merged_model_dir')).resolve()
        merged_prefix = best_chkpt_path.name.replace('checkpoint-', 'chkpt_')
        tuned_prefix = self._config.get('prefix', 'whisper_tune')
        merged_save_dir = merged_model_path.joinpath(f'{tuned_prefix}_{merged_prefix}')

        self._base_model_name = f'{self._config.get("model_name")}-{self._config.get("model_size")}'
        self._base_model_cache = Path(self._config.get('cache_dir', '~/.cache/huggingface/hub')).resolve()
        base_model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self._base_model_name,
            cache_dir=self._base_model_cache,
            device_map='auto',
        )

        tokenizer = AutoTokenizer.from_pretrained(self._base_model_name, cache_dir=self._base_model_cache)
        processor = AutoProcessor.from_pretrained(self._base_model_name, cache_dir=self._base_model_cache)
        peft_model = PeftModel.from_pretrained(base_model, best_chkpt_path)
        merged_model = peft_model.merge_and_unload()

        merged_model.save_pretrained(merged_save_dir)
        tokenizer.save_pretrained(merged_save_dir)
        processor.save_pretrained(merged_save_dir)
        logging.info(f'Merged "{best_chkpt_path.name}" into "{self._base_model_name}" and saved finetuned model to "{merged_save_dir}"')
        return merged_save_dir

    ##########################################################################################

    def _conver_to_ct2(self, model_dir: Path, base_model: bool = False) -> Path:
        c2_model_path = Path(self._config.get('ct2_model_dir', '~/ct2_whisper_models')).resolve()
        ct2_converted_path = c2_model_path.joinpath(self._base_model_name) if base_model else c2_model_path.joinpath(model_dir.name)

        ct2_command = 'uv run ct2-transformers-converter' \
                    f' --model {str(model_dir)}' \
                    f' --output_dir {str(ct2_converted_path)}' \
                    f' --quantization {self._config.get("quantization", 'float16')}'

        subprocess.run(ct2_command, shell=True, capture_output=True)
        logging.info(f'Saved ctranslate2 converted model to "{ct2_converted_path}"')
        return ct2_converted_path

    ##########################################################################################

    def _evaluate_converted_model(self, ct2_whisper_model_path: Path, dataset: Dataset, prefix: str) -> None:
        logging.info(f'Started to compute metrics for converted model "{ct2_whisper_model_path.name}"')
        ct2_whisper_model = Whisper(
            model_path=str(ct2_whisper_model_path),
            device='auto',
            compute_type=self._config.get('quantization', 'int8'),
        )
        whisper_processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path=self._base_model_name,
            cache_dir=self._base_model_cache,
            feature_size=self._feature_size,
        )
        evaluated_dataset = {
            'reference': [],
            'recognized': [],
            'wer': [],
            'cer': [],
        }
        total_wer = 0
        total_cer = 0
        for audio in tqdm(dataset, desc='Recognizing and calculating...', mininterval=20, maxinterval=30):
            recognized_text = _recognize_audio_file(
                file_path=audio['path'], 
                lang=self._config.get('language'),
                model=ct2_whisper_model,
                processor=whisper_processor,
            )
            sent_wer, sent_cer = _calculate_sent_wer_cer(hypothesis=recognized_text, reference=audio['sentence'])
            total_wer += sent_wer
            total_cer += sent_cer
            evaluated_dataset['reference'].append(audio['sentence'])
            evaluated_dataset['recognized'].append(recognized_text)
            evaluated_dataset['wer'].append(sent_wer * 100)
            evaluated_dataset['cer'].append(sent_cer * 100)

        average_cer = round(total_cer / len(evaluated_dataset), 2)
        average_wer = round(total_wer / len(evaluated_dataset), 2)
        logging.info(f'Average CER for "{ct2_whisper_model_path.name}": {average_cer}')
        logging.info(f'Average WER for "{ct2_whisper_model_path.name}": {average_wer}')

        if self._clearml_task:
            self._clearml_task.get_logger().report_single_value(
                name=f'CT2 {prefix} CER',
                value=average_cer,
            )
            self._clearml_task.get_logger().report_single_value(
                name=f'CT2 {prefix} WER',
                value=average_wer,
            )
            if prefix == 'Base':
                self._clearml_task.upload_artifact(
                    name=f'Sentences "{self._base_model_name}"',
                    artifact_object='\n'.join(evaluated_dataset['recognized']),
                )
            else:
                self._clearml_task.upload_artifact(
                    name='Sentences reference',
                    artifact_object='\n'.join(evaluated_dataset['reference']),
                )
                self._clearml_task.upload_artifact(
                    name=f'Sentences "{ct2_whisper_model_path.name}"',
                    artifact_object='\n'.join(evaluated_dataset['recognized']),
                )
        return evaluated_dataset

    ##########################################################################################

    def __call__(
        self,
        dataset: Dataset | None = None,
        best_chkpt_path: str | None = None,
        iter_step: int | None = None,
        adapters_path: str | None = None,
        iter_chkpt: bool = False,
        last_chkpt: bool = False,
    ) -> None:
        setup_environment(init=False)
        merged_model_path = self._merge_and_save(best_chkpt_path)
        ct2_whisper_model = self._conver_to_ct2(merged_model_path)
        setup_environment(init=False)
        prefix = 'Last' if last_chkpt else 'Best'
        evaluated_dataset = self._evaluate_converted_model(ct2_whisper_model, dataset, prefix=prefix)
        html_table_path = generate_html(evaluated_dataset, ct2_whisper_model.name)
        logging.info(f'Saved HTML to the file "{html_table_path}"')
        if self._clearml_task:
            self._clearml_task.get_logger().report_media(
                title='eval samples',
                series='Ct2 merged model test',
                iteration=iter_step,
                local_path=str(html_table_path),
                delete_after_upload=False,
            )
        if not last_chkpt:
            base_model_path = next(self._base_model_cache.rglob('model.safetensors')).parent
            base_ct2_whisper_model = self._conver_to_ct2(base_model_path, base_model=True)
            setup_environment(init=False)
            self._evaluate_converted_model(base_ct2_whisper_model, dataset, prefix='Base')
