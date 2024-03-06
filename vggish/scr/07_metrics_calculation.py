import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def silhouette_score_calc(input_csv, num_clusters):
    """
    Calculate silhouette score using KMeans clustering.

    Parameters:
    - input_csv (str): Path to the input CSV file containing embeddings.
    - num_clusters (int): Number of clusters to be used in KMeans clustering.

    Returns:
    - cluster_df (DataFrame): DataFrame containing identification numbers and their assigned cluster labels.
    """
    # Read the input DataFrame
    df = pd.read_csv(input_csv)

    # Extract identification numbers
    identification_numbers = df['identification_number']

    # Extract columns containing 2 dimensions
    dimension_columns = df.columns[1:]

    # Convert selected columns to a NumPy array
    data_array = df[dimension_columns].values

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=42)
    cluster_labels = kmeans.fit_predict(data_array)

    # Calculate silhouette score
    silhouette = silhouette_score(data_array, kmeans.labels_)

    # Create DataFrame with identification numbers and cluster labels
    cluster_df = pd.DataFrame({'identification_number': identification_numbers, 'cluster_number': cluster_labels})

    return cluster_df, silhouette

def known_confusion_matrix_calc(cluster_df, annotations_csv, output_heatmap):
    """
    Generate a heatmap showing percentage of labels within each cluster.

    Parameters:
    - cluster_df (DataFrame): DataFrame containing identification numbers and cluster labels.
    - annotations_csv (str): Path to the input CSV file containing annotations for the identification numbers.
    - output_heatmap (str): Path to save the output heatmap file.
    """
    # Read annotations DataFrame
    ann = pd.read_csv(annotations_csv)

    # Merge cluster DataFrame with annotations DataFrame
    merged_df = pd.merge(cluster_df, ann, how='left', left_on='identification_number', right_on='vggish_point')
    merged_df = merged_df.drop(columns=['vggish_point'], axis=1)
    merged_df = merged_df.dropna()

    # Count total number of each label
    cluster_count = merged_df['label'].value_counts().reset_index()
    cluster_count.columns = ['label', 'total_number']
    cluster_count = cluster_count.sort_values(by='label')

    # Count number of each label within each cluster
    cluster_label_counts = merged_df.groupby(['cluster_number', 'label']).size().reset_index(name='count')

    # Merge label counts with total counts
    cluster_merged = pd.merge(cluster_label_counts, cluster_count, how='left', on='label')
    cluster_merged['percentage'] = round((cluster_merged['count'] / cluster_merged['total_number']) * 100, 2)
    cluster_merged = cluster_merged.drop(['count', 'total_number'], axis=1)

    # Sort data and pivot for heatmap
    cluster_merged = cluster_merged.groupby('cluster_number').apply(lambda x: x.sort_values(['percentage'], ascending=False)).reset_index(drop=True)
    heatmap_data = cluster_merged.pivot("label", "cluster_number", "percentage")

    # Create heatmap
    plt.figure(figsize=(15, 8))
    heatmap = sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title('Heatmap of Percentage by Cluster and Label')
    plt.xlabel('Cluster Number')
    plt.ylabel('Label')

    heatmap.yaxis.set_tick_params(labelsize=6)

    # Save the heatmap as a file
    plt.savefig(output_heatmap)

    # Close the plot to prevent display
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the goodness of dimensionality reduction & clustering metrics")
    parser.add_argument("--input_csv", help="Input CSV file containing reduced embeddings from tsne or umap")
    parser.add_argument("--num_clusters", help="Number of clusters acquired from the dimensionality reduction process")
    parser.add_argument("--annotations_csv", help="Input CSV file containing the annotations for the identification numbers")
    parser.add_argument("--output_heatmap", help="Output heatmap file to save the heatmap")

    args = parser.parse_args()

    cluster_df, silhouette = silhouette_score_calc(args.input_csv, args.num_clusters)
    known_confusion_matrix_calc(cluster_df, args.annotations_csv, args.output_heatmap)

    print(f"Silhouette Score: {silhouette}")
