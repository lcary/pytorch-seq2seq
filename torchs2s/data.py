"""
Code from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Module for data preparation code.
"""
import logging
import os
import random
from io import open
# start/end of string
from typing import Tuple

from torchs2s.constants import MAX_LENGTH, Pairs, Pair
from torchs2s.language import Language, normalize_string

ENG_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

log = logging.getLogger(__name__)


def read_languages(lang1: str,
                   lang2: str,
                   reverse: bool = False) -> Tuple[Language, Language, Pairs]:
    log.info('Reading lines...')

    with open(os.path.join('data', '{}-{}.txt'.format(lang1, lang2)), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p: Pair) -> bool:
    """
    Filter method to trim data to short and simple sentences for training.
    """
    return (
            len(p[0].split(' ')) < MAX_LENGTH and
            len(p[1].split(' ')) < MAX_LENGTH and
            p[1].startswith(ENG_PREFIXES))


def filter_pairs(pairs: Pairs) -> Pairs:
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1: str,
                 lang2: str,
                 reverse: bool = False) -> Tuple[Language, Language, Pairs]:
    """
    1. Read text file and split into lines, split lines into pairs
    2. Normalize text, filter by length and content
    3. Make word lists from sentences in pairs
    """
    input_lang, output_lang, pairs = read_languages(lang1, lang2, reverse)
    log.info("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    log.info("Trimmed to %s sentence pairs" % len(pairs))
    log.info("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    log.info("Counted words:")
    log.info('{}, {}'.format(input_lang.name, input_lang.n_words))
    log.info('{}, {}'.format(output_lang.name, output_lang.n_words))
    return input_lang, output_lang, pairs


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    log.info(random.choice(pairs))
