import os
import random
import string
import unittest
from tempfile import TemporaryDirectory

from torch import Tensor

from torchs2s.evaluate import evaluate, evaluate_randomly, evaluate_and_save_attention
from tests.helpers import get_network_context


def random_str(str_len=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(str_len))


class TestEvaluation(unittest.TestCase):

    def test_evaluate_key_error(self):
        network_context = get_network_context()
        sentence = "he received a registered letter ."
        network_context.input_lang.add_sentence(sentence)
        with self.assertRaises(KeyError):
            evaluate(network_context, sentence, 10)

    def test_evaluate(self):
        network_context = get_network_context(output_size=1)
        sentence = "he received a registered letter ."
        network_context.input_lang.add_sentence(sentence)
        decoded_words, decoder_attentions = evaluate(network_context, sentence, 10)
        expect = ['SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS']
        self.assertEqual(decoded_words, expect)
        self.assertTrue(isinstance(decoder_attentions, Tensor))

    def test_evaluate_randomly(self):
        network_context = get_network_context(output_size=1)
        sentence = "he received a registered letter ."
        network_context.input_lang.add_sentence(sentence)
        pairs = [[sentence, "foo"]]
        rv = evaluate_randomly(network_context, pairs, 5)
        self.assertIsNone(rv)

    def test_evaluate_and_save_attention(self):
        network_context = get_network_context(output_size=1)
        sentence = "he received a registered letter ."
        network_context.input_lang.add_sentence(sentence)
        with TemporaryDirectory() as temp_dir:
            fp = os.path.join(temp_dir, "test.png")
            evaluate_and_save_attention(network_context, sentence, fp)
            self.assertTrue(os.path.exists(fp))


if __name__ == '__main__':
    unittest.main()
