import unittest
from unittest import mock

from torch import nn

from torchs2s.language import tensors_from_pair
from torchs2s.train import train_attention, train_iters
from tests.helpers import get_network_context


class TestTraining(unittest.TestCase):

    def test_train_attention(self):
        network_context = get_network_context()
        eng_sentence = "he received a registered letter ."
        fra_sentence = "il a recu une lettre recommandee ."
        pair = [eng_sentence, fra_sentence]
        network_context.input_lang.add_sentence(eng_sentence)
        network_context.output_lang.add_sentence(fra_sentence)
        pair = tensors_from_pair(network_context.input_lang, network_context.output_lang, pair)
        input_tensor = pair[0]
        output_tensor = pair[1]
        criterion = nn.NLLLoss()

        loss = train_attention(input_tensor, output_tensor, network_context, criterion)
        self.assertTrue(isinstance(loss, float))

    @mock.patch('torchs2s.train.save_plot')
    def test_train_iters(self, save_plot):
        network_context = get_network_context()
        eng_sentence = "he received a registered letter ."
        fra_sentence = "il a recu une lettre recommandee ."
        network_context.input_lang.add_sentence(eng_sentence)
        network_context.output_lang.add_sentence(fra_sentence)
        pairs = [[eng_sentence, fra_sentence]]
        train_iters(network_context, 10, pairs)
        self.assertTrue(save_plot.called)


if __name__ == '__main__':
    unittest.main()
