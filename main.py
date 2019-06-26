import logging

from torchs2s.constants import DEVICE
from torchs2s.data import prepare_data
from torchs2s.networks import EncoderRNN, AttentionDecoderRNN
from torchs2s.train import train_iters, evaluate_randomly

log = logging.getLogger('torchs2s')
log.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('torchs2s.log')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the log
log.addHandler(c_handler)
log.addHandler(f_handler)

hidden_size = 256
log.info('preparing data...')
input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

log.info('creating encoder RNN')
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
# attn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(DEVICE)

log.info('creating decoder RNN')
attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(DEVICE)

log.info('training for 75000 iterations')
train_iters(encoder1, attn_decoder1, 75000, input_lang, output_lang, pairs)

log.info('random evaluation for debugging')
evaluate_randomly(encoder1, attn_decoder1, input_lang, output_lang, pairs)

log.info('done.')
