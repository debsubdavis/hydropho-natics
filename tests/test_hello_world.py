"""
This module contains tests for the hello_world module.
"""

# The rest of your test code follows here...


from utils import hello_world

def test_get_message():
    """
    Test that get_message returns the correct string.
    """
    assert hello_world.get_message() == "Hello World"
