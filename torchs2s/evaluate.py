import random
from typing import List, Tuple

import torch
from torch import Tensor

from torchs2s.constants import MAX_LENGTH, DEVICE, SOS_token, EOS_token, Pairs
from torchs2s.graph import save_attention_matrix
from torchs2s.language import tensor_from_sentence
from torchs2s.networks import NetworkContext


def evaluate(context: NetworkContext,
             sentence: str,
             max_length: int = MAX_LENGTH) -> Tuple[List[str], Tensor]:
    encoder = context.encoder
    decoder = context.decoder
    input_lang = context.input_lang
    output_lang = context.output_lang

    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(context: NetworkContext, pairs: Pairs, n: int = 10) -> None:
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(context, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate_and_save_attention(context: NetworkContext,
                                input_sentence: str,
                                filename: str) -> None:
    output_words, attentions = evaluate(context, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    save_attention_matrix(input_sentence, output_words, attentions, filename)
