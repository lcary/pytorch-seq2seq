import logging

from torchs2s.constants import DEVICE
from torchs2s.data import prepare_data
from torchs2s.networks import EncoderRNN, AttentionDecoderRNN
from torchs2s.train import train_iters
from torchs2s.evaluate import evaluate_randomly, evaluate_and_save_attention
from torchs2s.utils import log_setup

log = log_setup()


def main():
    hidden_size = 256
    log.info('preparing data...')
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

    log.info('creating encoder RNN')
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
    # attn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(DEVICE)

    log.info('creating decoder RNN')
    attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(DEVICE)

    # n_iters = 75000
    n_iters = 100
    log.info('training for {} iterations'.format(n_iters))
    train_iters(encoder1, attn_decoder1, n_iters, input_lang, output_lang, pairs)

    log.info('random evaluation for debugging')
    evaluate_randomly(encoder1, attn_decoder1, input_lang, output_lang, pairs)

    evaluate_and_save_attention(encoder1, attn_decoder1, input_lang, output_lang, "elle a cinq ans de moins que moi .", 'sentence1.png')
    evaluate_and_save_attention(encoder1, attn_decoder1, input_lang, output_lang, "elle est trop petit .", 'sentence2.png')
    evaluate_and_save_attention(encoder1, attn_decoder1, input_lang, output_lang, "je ne crains pas de mourir .", 'sentence3.png')
    evaluate_and_save_attention(encoder1, attn_decoder1, input_lang, output_lang, "c est un jeune directeur plein de talent .", 'sentence4.png')

    log.info('done.')


if __name__ == '__main__':
    main()
