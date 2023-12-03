"""
A simple script to print "Hello World" that adheres to Pylint standards.
"""

import pandas as pd
import numpy as np
import pyspark

def get_message():
    """
    Returns the message to be printed.
    """
    return "Hello World"

def main():
    """
    Main function that prints the message returned by get_message().
    """
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    print("Pandas DataFrame:\n", data)

    array = np.array([1, 2, 3])

    array_sum = np.sum(array)
    print("Sum of numpy array:", array_sum)

    spark = pyspark.sql.SparkSession.builder.appName("HelloWorldApp").getOrCreate()
    print("Spark session created.")

    # Stop the Spark session
    spark.stop()

    message = get_message()
    print(message)

if __name__ == "__main__":
    main()
