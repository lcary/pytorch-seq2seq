import torch.optim as optim

from torchs2s.constants import DEVICE
from torchs2s.data import prepare_data
from torchs2s.evaluate import evaluate_randomly, evaluate_and_save_attention
from torchs2s.networks import EncoderRNN, AttentionDecoderRNN, NetworkContext
from torchs2s.train import train_iters
from torchs2s.utils import log_setup

log = log_setup()


def main() -> None:
    hidden_size = 256
    learning_rate = 0.01

    log.info('preparing data...')
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

    log.info('creating encoder RNN')
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
    # attn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(DEVICE)

    log.info('creating decoder RNN')
    decoder = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(DEVICE)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    context = NetworkContext(encoder, decoder, encoder_optimizer, decoder_optimizer, input_lang, output_lang)

    # n_iters = 75000
    n_iters = 100
    log.info('training for {} iterations'.format(n_iters))
    train_iters(context, n_iters, pairs)

    log.info('random evaluation for debugging')
    evaluate_randomly(context, pairs)

    evaluate_and_save_attention(context, "elle a cinq ans de moins que moi .", 'sentence1.png')
    evaluate_and_save_attention(context, "elle est trop petit .", 'sentence2.png')
    evaluate_and_save_attention(context, "je ne crains pas de mourir .", 'sentence3.png')
    evaluate_and_save_attention(context, "c est un jeune directeur plein de talent .", 'sentence4.png')

    log.info('done.')


if __name__ == '__main__':
    main()
