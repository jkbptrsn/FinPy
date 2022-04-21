import unittest


class TestSomething(unittest.TestCase):

    def test_This(self):
        self.assertEqual(2, 2)


if __name__ == '__main__':
    unittest.main()
