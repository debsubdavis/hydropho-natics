import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def silhouette_score_calc(input_csv, num_clusters):
    # read the input DataFrame
    df = pd.read_csv(input_csv)
    print("CSV file read")

    identification_numbers = df['identification_number']

    # Extract only the columns containing 2 dimensions
    dimension_columns = df.columns[1:]

    # Convert the selected columns to a NumPy array
    data_array = df[dimension_columns].values

    print("Starting KMeans clustering...")
    # Apply KMeans clustering with optimal number of clusters
    kmeans = KMeans(n_clusters = int(num_clusters), random_state = 42)
    cluster_labels = kmeans.fit_predict(data_array)
    print("KMeans clustering applied to the data")

    print(f"Silhoutte Score: {silhouette_score(data_array, kmeans.labels_)}")

    cluster_df = pd.DataFrame({'identification_number': identification_numbers, 'cluster_number': cluster_labels})

    return cluster_df

def known_confusion_matrix_calc(cluster_df, annotations_csv, output_heatmap):
    ann = pd.read_csv(annotations_csv)

    merged_df = pd.merge(cluster_df, ann, how='left', left_on='identification_number', right_on='vggish_point')
    merged_df = merged_df.drop(columns=['vggish_point'], axis = 1)
    merged_df = merged_df.dropna()
    print(f"Shape of the merged dataframe: {merged_df.shape}")

    # create a dataframe with two columns label and total number
    cluster_count = merged_df['label'].value_counts()
    cluster_count = cluster_count.reset_index()
    cluster_count.columns = ['label', 'total_number']
    cluster_count = cluster_count.sort_values(by='label')

    cluster_label_counts = merged_df.groupby(['cluster_number', 'label']).size().reset_index(name='count')

    cluster_merged = pd.merge(cluster_label_counts, cluster_count, how='left', left_on='label', right_on='label')
    cluster_merged['percentage'] = round((cluster_merged['count'] / cluster_merged['total_number']) * 100, 2)
    cluster_merged = cluster_merged.drop(['count', 'total_number'], axis = 1)

    # groupby cluster number and sort pecentage in descending order
    cluster_merged = cluster_merged.groupby('cluster_number').apply(lambda x: x.sort_values(['percentage'], ascending = False)).reset_index(drop=True)

    heatmap_data = cluster_merged.pivot("label", "cluster_number", "percentage")

    # Create the heatmap
    plt.figure(figsize=(15, 8))
    heatmap = sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title('Heatmap of Percentage by Cluster and Label')
    plt.xlabel('Cluster Number')
    plt.ylabel('Label')

    heatmap.yaxis.set_tick_params(labelsize=6)

    # Save the plot as a file
    plt.savefig(output_heatmap)
    print("Heatmap saved")

    # If you don't want to display the plot in the notebook or console, you can close it
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Reduce the dimensions of embeddings to 2D and save the result.")
    parser.add_argument("input_csv", help = "Input CSV file containing reduced embeddings from tsne or umap")
    parser.add_argument("num_clusters", help = "Number of clusters accquired from the dimensionality reduction process")
    parser.add_argument("annotations_csv", help = "Input CSV file containing the annotations for the identification numbers")
    parser.add_argument("output_heatmap", help = "Output heatmap file to save the heatmap")

    args = parser.parse_args()

    cluster_df = silhouette_score_calc(args.input_csv, args.num_clusters)
    known_confusion_matrix_calc(cluster_df, args.annotations_csv, args.output_heatmap)
