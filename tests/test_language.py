import unittest

from torch import Tensor

from torchs2s.language import Language, tensor_from_sentence, normalize_string, tensors_from_pair


class TestLanguage(unittest.TestCase):

    def test_tensor_from_sentence(self):
        sentence = "elle a cinq ans de moins que moi ."
        language = Language("eng")
        language.add_sentence(sentence)
        input_tensor = tensor_from_sentence(language, sentence)
        self.assertTrue(isinstance(input_tensor, Tensor))

    def test_tensor_from_sentence_key_error(self):
        sentence = "boo bah ."
        language = Language("eng")
        with self.assertRaises(KeyError):
            tensor_from_sentence(language, sentence)

    def test_normalize_string(self):
        in_str = "Il a reçu une lettre recommandée."
        out_str = normalize_string(in_str)
        expect = "il a recu une lettre recommandee ."
        self.assertEqual(out_str, expect)

    def test_tensors_from_pair(self):
        eng_sentence = "he received a registered letter ."
        fra_sentence = "il a recu une lettre recommandee ."
        pair = [eng_sentence, fra_sentence]
        eng_language = Language("eng")
        eng_language.add_sentence(eng_sentence)
        fra_language = Language("fra")
        fra_language.add_sentence(fra_sentence)
        out = tensors_from_pair(eng_language, fra_language, pair)
        self.assertTrue(isinstance(out, tuple))
        self.assertEqual(len(out), 2)
        self.assertTrue(isinstance(out[0], Tensor))
        self.assertTrue(isinstance(out[1], Tensor))


if __name__ == '__main__':
    unittest.main()
