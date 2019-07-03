import unittest

from torchs2s.data import prepare_data, filter_pairs, read_languages
from torchs2s.language import Language, normalize_string

DATA_FILE = "eng-fra.sample.txt"


class TestData(unittest.TestCase):

    def test_prepare_data(self):
        input_lang = 'eng'
        output_lang = 'fra.sample'
        input_lang, output_lang, pairs = prepare_data(input_lang, output_lang)
        self.assertTrue(isinstance(input_lang, Language))
        self.assertTrue(isinstance(output_lang, Language))
        self.assertEqual(len(pairs), 0)

    def test_read_languages_reverse(self):
        input_lang = 'eng'
        output_lang = 'fra.sample'
        input_lang, output_lang, pairs = read_languages(input_lang, output_lang, reverse=False)
        self.assertTrue(isinstance(input_lang, Language))
        self.assertTrue(isinstance(output_lang, Language))
        self.assertEqual(len(pairs), 20)

    def test_filter_pairs(self):
        pairs = [
            ["I'm hungry and thirsty.", "J'ai faim et soif."],
            ["he received a registered letter .", "il a recu une lettre recommandee ."],
            [
                (
                    "Food prices are at their highest level since the United Nations Food and "
                    "Agriculture Organization began keeping records in 1990."
                ),
                (
                    "Les prix de l'alimentation sont à leur plus haut niveau depuis que "
                    "l'Organisation des Nations Unies pour l’alimentation et l’agriculture "
                    "a commencé à les enregistrer en mille-neuf-cent-quatre-vingt-dix."
                )
            ],
            ["I'm no friend of yours.", "Je ne suis pas ton amie."],
            ["Who's he?", "Qui est-il ?"],
            ["I'm longing to see him.", "J'ai hâte de le voir."],
        ]
        pairs = [[normalize_string(i), normalize_string(j)] for (i, j) in pairs]
        pairs = [list(reversed(p)) for p in pairs]
        res = filter_pairs(pairs)
        self.assertEqual(len(res), 3)

    def test_read_languages(self):
        input_lang = 'eng'
        output_lang = 'fra.sample'
        input_lang, output_lang, pairs = read_languages(input_lang, output_lang)
        self.assertTrue(isinstance(input_lang, Language))
        self.assertTrue(isinstance(output_lang, Language))
        self.assertEqual(len(pairs), 20)


if __name__ == '__main__':
    unittest.main()
