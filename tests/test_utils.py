import unittest
from unittest.mock import patch

from torchs2s.utils import as_minutes, time_since


class TestUtils(unittest.TestCase):

    def test_as_minutes(self):
        mins = as_minutes(132.032)
        self.assertEqual(mins, "2m 12s")

    @patch('torchs2s.utils.time')
    def test_time_since(self, mock_time):
        mock_time.time.return_value = 1200
        out = time_since(1000, 1)
        self.assertEqual(out, '3m 20s (- 0m 0s)')


if __name__ == '__main__':
    unittest.main()
