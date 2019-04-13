import unittest

from gwmemory import harmonics


class TestHarmonics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fac(self):
        self.assertEqual(harmonics.fac(5), 120)
