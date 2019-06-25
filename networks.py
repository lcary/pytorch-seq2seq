"""
Code from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Module containing seq2seq networks.

A Recurrent Neural Network (RNN) operates on a sequence and uses its output
as input for subsequent steps.

A Sequence to Sequence (seq2seq) network, also known as an "Encoder Decoder"
network, is a model that consists of 2 RNNs. The first RNN is called the encoder,
which takes a sequence as input and outputs a vector. The vector encodes the
meaning of the input sequence into one vector in an N-dimensional space of
sentences. The second RNN is called the decoder, which reads the encoder's
output vector to produce an output sequence.

With a seq2seq network, each input sequence does not need to correspond to each
output. Sequence length and order can vary between inputs and output sequences.
For example, order and length vary in the input/output sequence of characters:

    "Je ne suis pas le chat noir" -> “I am not the black cat”

"""

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE, MAX_LENGTH


class EncoderRNN(nn.Module):
    """
    Outputs a vector and hidden state for each word of the input sentence.

    The hidden state is used for the next input word.

        input         previous_hidden
          |             |
          v             |
        embedding       |
          |             |
          v             v
        embedded -->  GRU
                      /  \
                     v   v
                output   hidden

    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Lookup table to store embeddings, which are mappings
        # from discrete variables to continuous ones:
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Multi-layer gated-recurrent unit: similar to long
        # short-term memory (LSTM) with forget gate:
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_data, hidden):
        embedded = self.embedding(input_data).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):
    """
    Simple decoder that uses the last output (aka the "context vector")
    of the encoder.

    The decoder receives an input token and hidden state at each step,
    with the start-of-string ("SOS") token used as the initial token.

        input         previous_hidden
          |             |
          v             |
        embedding       |
          |             |
          v             v
        ReLU  ------>  GRU
                      /  \
                     v   v
                   out   hidden
                    |
                    v
                  softmax
                    |
                    v
                  output
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(hidden_size, output_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Linear transformation of incoming data:
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        output = self.embedding(input_data).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
