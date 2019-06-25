from constants import DEVICE
from data import prepare_data
from networks import EncoderRNN, DecoderRNN, AttentionDecoderRNN
from train import train_iters, evaluate_randomly

hidden_size = 256
input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
# attn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(DEVICE)
attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(DEVICE)

train_iters(encoder1, attn_decoder1, 75000, input_lang, output_lang, pairs, print_every=5000)
evaluate_randomly(encoder1, attn_decoder1, input_lang, output_lang, pairs)
