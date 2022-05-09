import unittest


class TestSomething(unittest.TestCase):

    def test_this(self):
        print("test_template.py")
        self.assertEqual(2, 2)


if __name__ == '__main__':
    unittest.main()
