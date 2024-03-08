import argparse
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

def tsne_custom(input_csv, output_csv):
    """
    Perform t-SNE dimensionality reduction and DBSCAN clustering on embeddings.

    Parameters:
    - input_csv (str): Path to the input CSV file containing embeddings.
    - output_csv (str): Path to save the output CSV file with reduced embeddings.

    Returns:
    - None
    """
    # Read the input DataFrame
    df = pd.read_csv(input_csv)
    print("CSV file read")

    # Check for and drop 'Unnamed: 0' column if present
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        print("Dropped 'Unnamed: 0' column")
    else:
        pass

    # Extract 'identification_number' column
    identification_numbers = df['identification_number']

    # Extract columns containing 128 dimensions
    dimension_columns = df.columns[:-1]

    # Convert selected columns to a NumPy array
    data_array = df[dimension_columns].values

    print("Dataframe converted to a 2D Numpy Array")
    print(f'The length of the array is: {len(data_array)}')

    # Apply t-SNE for dimensionality reduction to 2D
    reducer = TSNE(n_components=2, perplexity=50, random_state=42)
    reduced_embeddings = reducer.fit_transform(data_array)
    print("t-SNE applied to the data")

    # Apply DBSCAN for clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(reduced_embeddings)

    # Count the number of clusters (excluding noise points, labeled as -1)
    n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters: {n_clusters}")

    # Create a DataFrame with 'identification_number', 'x', and 'y' columns
    result_df = pd.DataFrame({
        'identification_number': identification_numbers,
        'x': reduced_embeddings[:, 0],  # Extract the first dimension
        'y': reduced_embeddings[:, 1],  # Extract the second dimension
    })

    # Save the result DataFrame to the output CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"Reduced embeddings saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce the dimensions of embeddings to 2D and save the result.")
    parser.add_argument("--input_csv", help="Input CSV file containing embeddings")
    parser.add_argument("--output_csv", help="Output CSV file to save the reduced embeddings")

    args = parser.parse_args()

    tsne_custom(args.input_csv, args.output_csv)
