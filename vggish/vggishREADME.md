## About
This README contains information on the VGGish model, how to resample audio, run VGGish, and cluster the output embeddings.

## Steps to Execute Analysis

### I. Set up environment and verify VGGish runs via a "smoke test"
1. From the vggish directory, create and activate the VGGish virtual environment using the below code:
    - $ conda env create -f environment.yml 
    - $ conda activate vggish_run_environment
2. Navigate to vggish/scr and run the following code to install dependent libraries. They should already be included in the VGGish environment.yml file, but this will cover you in the event that they are not.
    - $ pip install -r requirements.txt
3. VGGish also requires downloading two data files. Download the below files and save them to the VGGish/scr directory.
    - [VGGish model checkpoint](https://storage.googleapis.com/audioset/VGGish_model.ckpt),
    in TensorFlow checkpoint format.
    - [Embedding PCA parameters](https://storage.googleapis.com/audioset/VGGish_pca_params.npz),
    in NumPy compressed archive format.
4. Execute the smoke test script from the VGGish/scr directory using the commands below.
    - $ python 01_vggish_smoke_tst.py
    - If you see a series of messages ending in "Looks Good To Me", you're all set.
    - If you don't get the "Looks Good to Me" message, interpret the error code and/or try following the steps from the VGGish [README](https://github.com/tensorflow/models/blob/master/research/audioset/VGGish/README.md). A common reason you may get the "tensorflow has no python module" error is because you aren't in the vggish_run_environment (or didn't pip install the requirements.txt).
    - Each time you run the model it will throw several depreciation warnings. These are due to an updated version of TensorFlow which the model has not been made compatible with. The libraries in the environment.yml file have specific versions noted to try to ensure compatability.


### II. Gather and resample audio files for embeddings
1. Create a directory containing all your audio files for embedding. They should all be in .wav file format.
2. VGGish was trained (and assumes) all audio is sampled as 16kHz mono. If your data doesn't come in this format, please run the 02_audio_resampling.py code in the scr directory using the code below. You will need to input the file path to your raw wav file directory created above and your desired save path for the resampled audio. The save path should be different from the wav path. The VGGish model will search for all .wav files in the given path, meaning if the save and wav path are the same, the model will try to embed both the raw and resmpled audio files. Becaue the 02_audio_resampling code creates new copies of the .wav files you should verify you have sufficient storage space before running it.
      $ python 02_audio_resampling.py --wav_path path/to/wav/files/ --save_path path/to/resampled/wav/files/
Resampled wav files will retain their original name with '_resampled' added to the end. This was done to allow researchers to easily distinguish between raw and resampled wav files.


### III. Create Audio Embeddings
Per the TensorFlow VGGish documentation:
"VGGish was trained with audio features computed as follows:
    *All audio is resampled to 16 kHz mono.
    *A spectrogram is computed using magnitudes of the Short-Time Fourier Transform with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann window.
    *A mel spectrogram is computed by mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz.
    *A stabilized log mel spectrogram is computed by applying log(mel-spectrum + 0.01) where the offset is used to avoid taking a logarithm of zero.
    *These features are then framed into non-overlapping examples of 0.96 seconds, where each example covers 64 mel bands and 96 frames of 10 ms each."
Sometimes a resesarcher may want to create an example (embedding) representing more or less than 0.96 seconds of audio. This can be achieved by changing the STFT_WINDOW_LENGTH_SECONDS, STFT_HOP_LENGTH_SECONDS, EXAMPLE_WINDOW_SECONDS, and EXAMPLE_HOP_SECONDS variables in the scr/vggish_params.py file. Begin by changing the EXAMPLE_WINDOW_SECONDS and EXAMPLE_HOP_SECONDS ("hop" is the overlap between examples - our tests used 0 overlap, meaing that EXAMPLE_WINDOW_SECONDS = EXAMPLE_HOP_SECONDS). The EXAMPLE_WINDOW_SECONDS / STFT_HOP_LENTGH_SECONDS must equal 96 due to the 96 frames noted in the model documentation quoted above. Alter the STFT_HOP_LENGTH_SECONDS to ensure EXAMPLE_WINDOW_SECONDS divided by it equals 96. STFT_HOP_LENGTH_SECONDS typically equals STFT_WINDOW_LENGTH_SECONDS * 0.4 (there's typically a 40% overlap when calculating the Fast Fourier Transform). By altering these 4 parameters you can continue to use the program as written while changing the amount of time represented in each example/embedding.

Formulaic value creation:
EXAMPLE_WINDOW_SECONDS: x
EXAMPLE_HOP_SECONDS: x (assuming 0 overlap between examples)
STFT_WINDOW_LENGTH_SECONDS: (x/96)/0.4 (assuming 40% overlap for Fast Fourier Transform)
STFT_HOP_LENGTH_SECONDS: x/96

Inputs for 0.96 second embeddings, 40% overlap:
EXAMPLE_WINDOW_SECONDS: 0.96
EXAMPLE_HOP_SECONDS: 0.96
STFT_WINDOW_LENGTH_SECONDS: 0.025
STFT_HOP_LENGTH_SECONDS: 0.010

Inputs for 2 second embeddings, 40% overlap:
EXAMPLE_WINDOW_SECONDS: 2
EXAMPLE_HOP_SECONDS: 2
STFT_WINDOW_LENGTH_SECONDS: 5/96
STFT_HOP_LENGTH_SECONDS: (1/48)

Inputs for 5 second embeddings, 40% overlap:
EXAMPLE_WINDOW_SECONDS: 5
EXAMPLE_HOP_SECONDS: 5
STFT_WINDOW_LENGTH_SECONDS: 25/192
STFT_HOP_LENGTH_SECONDS: 5/96

1. Ensuring that your VGGish_environment is activated, run the 03_vggish_audio_embeddings.py code as below:
      $ python 03_vggish_audio_embeddings.py --wav_path path/to/wav/files/ --save_path path/to/where/embeddings/should/go/
Audio embeddings will be saved as csv files (1 csv per wav file) in the specified save_path directory. No post-processing will be done on the data. This code can take a few seconds to run per audio file, meaning that many files may take a few hours. The code is not designed to run in any kind of parallel processing setup and would require significant modifications to do so due to how data is parsed to create the log mel spectrograms for embedding (see the documentation for stride_tricks, a library used in the frame() function of mel_features.py for more technical details).


### IV. Perform dimensionality reduction and clustering
1. To cluster the embeddings and identify like sounds we first need to combine all CSV embedding files output by VGGish. Do so programmatically by running the following command from the scr directory:
      $ python 04_combine_csv_files.py --folder_path path/to/vggish/output/csvs/ --output_file_path file/path/combined_csv_name.csv
2. Next we preprocess the combined embedding file. Each embedding is given a unique identifier based on its wav file name and sequential order e.g., the first N second embeding from the first wav file would be 1-0. The following command is run from the scr directory.
      $ python 05_preprocess_embeddings.py --input_csv path/to/combined/embedding/csv/combined_csv_name.csv --output_csv path/to/preprocessed/embeddings/preprocessed_csv_name.csv --mapping_csv /path/to/output/mapping/file/mapping_file.csv
3. We tested two methods of dimensionality reduction to take the embeddings from 128-D to 2-D for visuaslization and clustering purposes. t-SNE had the best results based on our metrics (see step 6 below). The t-SNE script can be run using the following command from the scr directory. It outputs 1 row per embedding with the id number and x and y coordinate:
      $ python 06_tsne_process.py --input_csv path/to/preprocessed/embeddings/preprocessed_csv_name.csv --output_csv path/to/reduced/embeddings/tsne_csv.csv
If you would like to run UMAP dimensionality reduction rather than t-SNE on the 128-D embeddings, run the following command from the scr directory:
      $ python umap_process.py --input_csv path/to/preprocessed/embeddings/preprocessed_csv_name.csv --output_csv path/to/reduced/embeddings/umap_csv.csv

*Optional: As part of the 2024 MSDS Capstone project we were given spectrogram annotations stored in a JSON format corresponding to some of the wav files. We used these annotations to label a subset of VGGish embeddings with their annotated sounds. We have included copies of the csvs necessary to execute annotation to embedding mappings in the top level utils folder without rerunning the code (unique_image_annotations.csv, human_readable_annotation_timings.csv, and vggish_0.96_sec_comb_labels.csv). Most researchers likely won't have access to the specific folders required to run the following scripts, but we include the code and steps here as a jumping off point should they wish to do a similar analysis.*

4. Labeling embeddings with known sounds requires running several scripts. These scripts have been specifically tailored to the structure of the hard drive passed to the 2024 MSDS Capstone team. If you have the same hard drive/data structure, start by running all cells in unique_images_annotations.ipynb found in the top level utils directory. This code will output unique_images_annotations.csv which contains only the most recent annotations for each spectrogram image (some images were annotated more than once). We have saved unique_images_annotations.csv in the utils folder so you don't have to run the code to achieve our results.
5. Next, run the embedding_to_annotation_mapper.ipynb. This will take the unique spectrogram annotations from step 4 and map those known sounds to embedding identification numbers based on the model parameters stored in vggish_params.py. If vggish_params.py is set up to create 0.96 second embeddings, the embedding_to_annotations_mapper will produce 0.96 second embedding mappings. If vggish_params.py is set up to create 2 second embeddings, the embedding_to_annotations_mapper will produce 2 second embedding mappings, and so on. We have saved several such embedding files, vggish_0.96_sec_comb_labels.csv, vggish_2_sec_comb_labels.csv and vggish_5_sec_comb_labels.csv so our results can be replicated without running the code.
6. We used Silhouette score and a custom "density of known sounds per cluster" metric to assess goodness of dimensionality reduction and clustering. More information on the density of known sounds per cluster can be found in the VGGish_report file in the vggish directory. To calculate the Silhouette score and create a heatmap with the density of known sounds, run the following command from the scr directory. The number of clusters was printed to the terminal when you ran tSNE or UMAP.
      $ python 07_metrics_calculation.py --input_csv path/to/reduced_embeddings.csv --num_clusters N --annotations_csv path/to/annotation.csv --output_heatmap path/to/save/heatmap/heatmap.png
*Note: 2 or more clusters must be found in t-SNE or UMAP for the Silhouette Score and sound density code to run properly*
7. To visualize the final clustering of sounds in 2-D, please run all cells in 08_final_tsne_plot.ipynb. Note that users will need to input the path to the reduced embeddings and embedding annotations.




## Data Files Created

### Resampled .wav files
*_resampled.wav - These wav files will be created in the user specified directory after running 02_audio_resampling.py. They will have the same names as the original wav files with '_resampled' appended to the end. They will be sampled as signed 16-bit PCM, 16kHz mono.

### Embedding CSV Files
*_resampled.csv - These files are generated by running 03_vggish_audio_embeddings.py. There is one csv file per wav file fed through the VGGish model. These csv files contain the 128-dimensional embeddings output from the VGGish model (first 128 columns) plus the name of the wav file, example_number, start time in seconds of the example, and end time in seconds for the example.

### Combined Embedding CSV File
[custom name].csv - Generated by 04_combine_csv_files.py. One csv file combining all embeddings generated for all wav files through VGGish. It has the same column structure as the original files: 128-dimensional embeddings output from the VGGish model (first 128 columns) plus the name of the wav file, example_number, start time in seconds of the example, and end time in seconds for the example.

### Cleaned Combined Embedding CSV File
[custom name].csv - Generated by 05_preprocess_embeddings.py. One csv file with all the combined VGGish embeddings. The wav filename, example number, and start/stop time columns have been removed. The last column is "identification_number" which contains the unique numerical identifier for each embedding. The first digit corresponds to the wav file and the digit after the '-' corresponds to the example number. Wav files are 1-indexed, embeddings are 0-indexed.

### Mapping CSV File
[custom name].csv - Generated by 05_preprocess_embeddings.py. This file maps each wav file to a unique number. This is done to make embedding labels shorter. The wav file number is used in the embedding identification_number.

### Reduced Combined Embedding CSV File
#### t-SNE
[custom name].csv - Generated by 06_tsne_process.py. Contains the t-SNE dimensionally reduced 2-D embeddings. Each row has the embedding's identification_number, x coordinate, and y coordinate.

#### UMAP
[custom name].csv - Generated by umap_process.py. Contains the UMAP dimensionally reduced 2-D embeddings. Each row has the embedding's identification_number, x coordinate, and y coordinate.



