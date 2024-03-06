import argparse
import pandas as pd

def preprocess(input_csv, output_csv, mapping_csv):
    """
    Preprocesses the input CSV file containing embeddings.

    Args:
    input_csv (str): Path to the input CSV file containing embeddings.
    output_csv (str): Path to the output CSV file to save the cleaned embeddings.
    mapping_csv (str): Path to the mapping CSV file to save the WAV file to identification number mapping.
    """

    # Read the input CSV file into a DataFrame
    data = pd.read_csv(input_csv)
    print("Data read from the input CSV file")

    # Get unique filenames from the input data and create a mapping from filename to identifier
    unique_filenames = data['wav_filename'].unique()
    filename_mapping = {filename: i+1 for i, filename in enumerate(unique_filenames)}

    # Create a DataFrame to store the mapping between WAV filenames and identification numbers
    mapping_df = pd.DataFrame({'wav_filename': unique_filenames, 'mapped_filename': range(1, len(unique_filenames) + 1)})
    print("Mapping DataFrame created")

    # Merge the mapping DataFrame with the input DataFrame based on 'wav_filename' column
    df = data.merge(mapping_df, on='wav_filename', how='left')

    # Create a new column 'identification_number' by combining 'mapped_filename' and 'example_number'
    df['identification_number'] = df['mapped_filename'].astype(str) + '-' + df['example_number'].astype(str)

    # Save the mapping DataFrame to the mapping CSV file
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"Mapping file saved")

    # Drop unnecessary columns from the DataFrame
    cleaned_df = df.drop(['wav_filename', 'example_number', 'recording_start_s', 'recording_stop_s', 'mapped_filename'], axis=1)

    # Save the cleaned DataFrame to the output CSV file
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Cleaned embeddings saved")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess the input CSV file containing embeddings.")
    parser.add_argument("input_csv", help="Input CSV file containing embeddings")
    parser.add_argument("output_csv", help="Output CSV file to save the cleaned embeddings")
    parser.add_argument("mapping_csv", help="Mapping CSV file to save the WAV file to identification number mapping")

    args = parser.parse_args()

    # Call the preprocess function with the provided arguments
    preprocess(args.input_csv, args.output_csv, args.mapping_csv)

