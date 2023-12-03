"""
A simple script to print "Hello World" that adheres to Pylint standards.
"""

def get_message():
    """
    Returns the message to be printed.
    """
    return "Hello World"

def main():
    """
    Main function that prints the message returned by get_message().
    """
    message = get_message()
    print(message)

if __name__ == "__main__":
    main()
