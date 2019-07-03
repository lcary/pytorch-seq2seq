import os
import random
import string
import unittest
from tempfile import TemporaryDirectory

from torch import tensor

from torchs2s.graph import save_attention_matrix, save_plot


def random_str(str_len=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(str_len))


class TestEvaluation(unittest.TestCase):

    def test_save_plot(self):
        points = [
            [0.1031, 0.1136, 0.0435, 0.0643, 0.1319, 0.0592, 0.1732, 0.1145, 0.0980, 0.0986],
            [0.0945, 0.1031, 0.0590, 0.0808, 0.1279, 0.0485, 0.1846, 0.1298, 0.0956, 0.0762]]
        with TemporaryDirectory() as temp_dir:
            fp = os.path.join(temp_dir, "test.png")
            save_plot(points, filename=fp)
            self.assertTrue(os.path.exists(fp))

    def test_save_attention_matrix(self):
        sentence = "he received a registered letter ."
        output_words = ['SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS']
        attentions = tensor([
            [0.1031, 0.1136, 0.0435, 0.0643, 0.1319, 0.0592, 0.1732, 0.1145, 0.0980, 0.0986],
            [0.1194, 0.1168, 0.0401, 0.0854, 0.1188, 0.0539, 0.1607, 0.1206, 0.0806, 0.1037],
            [0.1338, 0.0701, 0.0545, 0.0734, 0.1058, 0.0729, 0.1599, 0.1356, 0.1151, 0.0789],
            [0.0987, 0.1152, 0.0558, 0.0616, 0.1261, 0.0672, 0.1792, 0.1262, 0.0913, 0.0787],
            [0.1098, 0.1247, 0.0541, 0.0884, 0.1199, 0.0566, 0.1357, 0.1122, 0.1185, 0.0801],
            [0.0995, 0.0838, 0.0723, 0.0735, 0.1393, 0.0608, 0.1244, 0.1405, 0.1073, 0.0987],
            [0.1276, 0.0954, 0.0665, 0.0659, 0.1379, 0.0549, 0.1392, 0.1252, 0.0895, 0.0979],
            [0.1184, 0.1056, 0.0514, 0.0873, 0.1111, 0.0576, 0.1356, 0.1364, 0.0958, 0.1007],
            [0.1189, 0.1092, 0.0594, 0.0718, 0.1208, 0.0549, 0.1481, 0.1167, 0.1262, 0.0740],
            [0.0945, 0.1031, 0.0590, 0.0808, 0.1279, 0.0485, 0.1846, 0.1298, 0.0956, 0.0762]])
        with TemporaryDirectory() as temp_dir:
            fp = os.path.join(temp_dir, "test.png")
            save_attention_matrix(sentence, output_words, attentions, fp)
            self.assertTrue(os.path.exists(fp))


if __name__ == '__main__':
    unittest.main()
