"""
This module contains tests for the hello_world module.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../")) #puts us in the hydropho-natics directory
from utils import hello_world

def test_get_message():
    """
    Test that get_message returns the correct string.
    """
    assert hello_world.get_message() == "Hello World"
