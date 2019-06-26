import logging
import random
import time

import torch
from torch import nn

from torchs2s.constants import SOS_token, DEVICE, EOS_token, MAX_LENGTH
from torchs2s.graph import save_plot
from torchs2s.utils import time_since
from torchs2s.language import tensors_from_pair

TEACHER_FORCING_RATIO = 0.5

log = logging.getLogger(__name__)


# TODO: get working with simple DecoderRNN
#
# current error: RuntimeError: input.size(-1) must be equal to input_size. Expected 256, got 2803
# when using: decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#
def train_attention(input_tensor, target_tensor, context, criterion, max_length=MAX_LENGTH):
    encoder = context.encoder
    decoder = context.decoder
    encoder_optimizer = context.encoder_optimizer
    decoder_optimizer = context.decoder_optimizer

    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(context, n_iters, pairs, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    input_lang = context.input_lang
    output_lang = context.output_lang

    training_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs))
                      for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    log.info('initializing training...')
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_attention(input_tensor, target_tensor, context, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter in [0, 1] or iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            log.info('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    save_plot(plot_losses)
