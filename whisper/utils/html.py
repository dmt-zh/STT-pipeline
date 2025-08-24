import difflib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template

import regex

##############################################################################################

def _make_html_table_row(
    reference: str = 'Human recognition',
    hypothesis: str = 'Model recognition',
    cer_metric: str = 'CER',
    wer_metric: str = 'WER',
    header: bool = False,
    model_name: str | None = None,
) -> str:
    tag_type = 'h' if header else 'd'
    table_values = (
        '<h2 align="center">Evaluation of ',
        f'{model_name}</h2>',
    )
    table_style = (
        '<style>\n\tbody {\n\ttable {margin: 0 auto; width: 100%;}\n',
        '\ttable, th, td {border-collapse: collapse;}\n',
        '\tth {position: sticky; top: 0; background-color: #fff;}\n\t}\n</style>\n',
    )
    table_header = '\n<table border="1" align="center">\n'

    table_row_pattern = (
        "\t<tr>\n",
        f'\t\t<t{tag_type}><font face="Calibri">{reference}</t{tag_type}>\n',
        f'\t\t<t{tag_type}><font face="Calibri">{hypothesis}</t{tag_type}>\n',
        f'\t\t<t{tag_type} align="center"><font face="Calibri">{cer_metric}</t{tag_type}>\n',
        f'\t\t<t{tag_type} align="center"><font face="Calibri">{wer_metric}</t{tag_type}>\n',
        '\t</tr>\n'
    )
    if header:
        return ''.join([''.join(table_style), ''.join(table_values), table_header, ''.join(table_row_pattern)])
    return ''.join(table_row_pattern)

##############################################################################################

def _paste_string(sentence: str, diff_words: Sequence[str], color: str) -> str:
    if diff_words:
        pattern = Template('<span style="background-color:$color">$word</span>')
        reversed_seq = [(idx, word) for idx, word in enumerate(diff_words)][::-1]
        for idx, word in reversed_seq:
            formatted_part = pattern.substitute(color=color, word=word)
            try:
                seq_detected = regex.search(fr'(?<!>){regex.escape(word)}(?=[^<]|$)', sentence, pos=idx)
                seq_start, _ = seq_detected.span()
                sent_start = sentence[:seq_start]
                sent_end = sentence[seq_start:]
                sentence = sent_start + regex.sub(fr'\b{word}\b', formatted_part, sent_end, count=1)
            except AttributeError:
                sentence = regex.sub(fr'\b{word}\b', formatted_part, sentence, count=1)
    return sentence

##############################################################################################

def _span_wrapper(reference: str, hypothesis: str) -> tuple[str]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    delta = tuple(difflib.Differ().compare(ref_words, hyp_words))
    diff_ref = tuple(word[1:].strip() for word in delta if word.startswith('-'))
    diff_hyp = tuple(word[1:].strip() for word in delta if word.startswith('+'))
    wrapped_ref = _paste_string(reference, diff_ref, '#5CE046')
    wrapped_hyp = _paste_string(hypothesis, diff_hyp, '#FFB4A8')
    return wrapped_ref, wrapped_hyp

##############################################################################################

def generate_html(dataset: Mapping[str, Sequence[str | float]], model_name: str) -> Path:
    whisper_parent_dir = Path(__file__).parent.parent
    save_html_dir = whisper_parent_dir.joinpath('htmls')
    save_html_dir.mkdir(exist_ok=True, parents=True)
    save_html_path = save_html_dir.joinpath(f'{model_name}.html')
    logging.info('Generating HTML table with predictions for final model.')

    with open(save_html_path, 'w', encoding='utf8') as html_table:
        html_table_rows = []
        html_table.write(_make_html_table_row(header=True, model_name=model_name))
        for ref_line, hyp_line, cer_line, wer_line in zip(
            dataset.get('reference'),
            dataset.get('recognized'),
            dataset.get('wer'),
            dataset.get('cer'), strict=False,
        ):
            pasted_ref, pasted_hyp = _span_wrapper(ref_line, hyp_line)
            table_row = _make_html_table_row(
                reference=pasted_ref,
                hypothesis=pasted_hyp,
                cer_metric=str(round(cer_line, 2)).replace('.', ','),
                wer_metric=str(round(wer_line, 2)).replace('.', ','),
            )
            html_table_rows.append((table_row, cer_line))
        for row in sorted(html_table_rows, key=lambda x: x[-1], reverse=True):
            html_table.write(row[0])
        html_table.write('</table>')
    return save_html_path
