"""
This module contains tests for the hello_world module.
"""

import sys
import os
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class TestHelloWorld(unittest.TestCase):
    """
    Test suite for hello_world.py
    """
    def test_message(self):
        """
        Test that get_message returns the correct string
        """
        from utils import hello_world
        self.assertEqual(hello_world.get_message(), "Hello World")


if __name__ == '__main__':
    unittest.main()
