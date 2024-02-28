import os
import argparse
import pandas as pd
from tqdm import tqdm

def combine_csv_files(folder_path, output_file):
    """
    Combines multiple CSV files into a single CSV file.

    Args:
    folder_path (str): Path to the folder containing CSV files.
    output_file (str): Output file to save the combined CSV data.
    """

    # Get a list of CSV files in the specified folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Check if there are any CSV files in the folder
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in tqdm(csv_files, desc="Combining CSV files"):
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined data to the output CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combine multiple CSV files into a single CSV file.")
    parser.add_argument("folder_path", help="Path to the folder containing CSV files")
    parser.add_argument("output_file", help="Output file to save the combined CSV data")
    args = parser.parse_args()

    # Call the function to combine CSV files
    combine_csv_files(args.folder_path, args.output_file)


