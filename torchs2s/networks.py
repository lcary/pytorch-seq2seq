"""
Code from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Module containing seq2seq networks.
"""

from __future__ import unicode_literals, print_function, division

from typing import Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer

from torchs2s.constants import DEVICE, MAX_LENGTH
from torchs2s.language import Language


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

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Lookup table to store embeddings, which are mappings
        # from discrete variables to continuous ones:
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Multi-layer gated-recurrent unit: similar to long
        # short-term memory (LSTM) with forget gate:
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, *inputs: Tensor) -> Tuple[Tensor, Tensor]:
        input_data = inputs[0]
        hidden = inputs[1]
        embedded = self.embedding(input_data).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self) -> Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


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

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(hidden_size, output_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Linear transformation of incoming data:
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, *inputs: Tensor) -> Tuple[Any, Tensor]:
        input_data = inputs[0]
        hidden = inputs[1]
        output = self.embedding(input_data).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self) -> Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class AttentionDecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 dropout_p: float = 0.1,
                 max_length: int = MAX_LENGTH) -> None:
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

    def forward(self, *inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        input_data = inputs[0]
        hidden = inputs[1]
        encoder_outputs = inputs[2]

        embedded = self.embedding(input_data).view(1, 1, -1)
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

    def init_hidden(self) -> Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class NetworkContext(object):
    """
    Context object for passing references to the different components of
    the network architecture. Mainly used to avoid long method signatures.
    """

    def __init__(self,
                 encoder: EncoderRNN,
                 decoder: DecoderRNN,
                 encoder_optimizer: Optimizer,
                 decoder_optimizer: Optimizer,
                 input_lang: Language,
                 output_lang: Language) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.input_lang = input_lang
        self.output_lang = output_lang
