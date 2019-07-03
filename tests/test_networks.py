import os
import unittest
from tempfile import TemporaryDirectory

import torch
import torch.optim as optim
from torch import Tensor, nn

from torchs2s.constants import DEVICE, SOS_token
from torchs2s.language import tensor_from_sentence, Language
from torchs2s.networks import EncoderRNN, AttentionDecoderRNN
from tests.helpers import get_network_context


class TestNetworks(unittest.TestCase):

    def test_encoder_step(self):
        encoder = EncoderRNN(1000, 256).to(DEVICE)
        encoder_hidden = encoder.init_hidden()
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)

        sentence = "elle a cinq ans de moins que moi ."
        language = Language("fra")
        language.add_sentence(sentence)
        input_tensor = tensor_from_sentence(language, sentence)
        input_length = input_tensor.size(0)

        encoder_optimizer.zero_grad()
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
        encoder_optimizer.step()

        self.assertTrue(isinstance(encoder_output, Tensor))
        self.assertTrue(isinstance(encoder_hidden, Tensor))

    def test_attention_decoder_step(self):
        encoder = EncoderRNN(1000, 256).to(DEVICE)
        decoder = AttentionDecoderRNN(256, 1000, dropout_p=0.1).to(DEVICE)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
        decoder_hidden = encoder.init_hidden()

        eng_sentence = "he seemed surprised at the news ."
        eng = Language("eng")
        eng.add_sentence(eng_sentence)
        target_tensor = tensor_from_sentence(eng, eng_sentence)
        target_length = target_tensor.size(0)

        loss = 0
        criterion = nn.NLLLoss()
        encoder_outputs = torch.zeros(10, 256, device=DEVICE)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

        loss.backward()  # type: ignore

        decoder_optimizer.step()

        self.assertTrue(isinstance(loss, Tensor))
        self.assertTrue(isinstance(decoder_output, Tensor))
        self.assertTrue(isinstance(target_tensor, Tensor))

    def test_network_context_state(self):
        context = get_network_context()
        with TemporaryDirectory() as temp_dir:
            state_file = os.path.join(temp_dir, 'test.state')
            context.save_state(state_file)
            self.assertTrue(os.path.exists(state_file))
            try:
                context.load_state(state_file)
            except Exception:
                self.fail('Unable to load saved network states.')


if __name__ == '__main__':
    unittest.main()
