This README contains information to run the test VGGish implementation

## About
For the initial POC we wanted to create embeddings for some test wav files and examine the PCA-reduced clusters visually.

Limitations and concerns include: small number of observations available per sound-type (e.g., lots of "dead sound" time), the reduction from 128 dimensions (number of dimensions created from the embedding) to 2 or 3 dimensions may not accurately reflect the distance of the clusters in space, model training data may not port well to hydrophone sounds.


## Steps to Reproduce Analysis

### Setting up and Testing VGGish
1. Activate the vggish_environment virtual environment
2. Download the "requirements.txt" file from https://github.com/tensorflow/models/tree/master/research/audioset/vggish and move it into the directory you plan to run your code in. For us it's hydropho-natics/poc/vggish.
3. Run "pip install -r requirements.txt" which installs the dependent libraries. These libraries should be contained in the vggish_environemnt file, but this will cover you in the event that they are not.
4. *Taken from the TensorFlow VGGish README linked above*
    - Clone TensorFlow models repo into a 'models' directory.
    - $ git clone https://github.com/tensorflow/models.git
    - $ cd models/research/audioset/vggish
    - Download data files into same directory as code.
    - $ curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
    - $ curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
5. Execute the smoke test *Taken from the TensorFlow VGGish README linked above*
    - Installation ready, let's test it.
    - $ python vggish_smoke_test.py
    - If we see "Looks Good To Me", then we're all set.

### Getting WAV files for analysis
1. Copy the JSON annotation files from the external hard drive/Annotaion Stuff/MLFigs_Labeled_Oct_26 into raw_data/. We will use these annotated files for our intial POC.

2. Run the metadata_file_finder to get the list of metadata files we need to copy from the external hard drive into the raw_data/ folder. Copy the files returned by the program into the raw_data folder. (Hint: they are '20181227T053004-File-10Metadata','20181227T100004-File-20Metadata', '20181209T083004-File-21Metadata', '20181204T203004-File-0Metadata', '20181227T100004-File-6Metadata'). Please see the metadata_file_finder file for information on why we chose these files specifically.

3. Open each of the metadata files to obtain the FileNames. The first numerical part corresponds to their hydrophone recordings. The matches are shown below:
    Sound        Metadata File Name                FileName                                              Hydrophone Recording Name
    Moorings     20181204T203004-File-0Metadata    181204-203002-437599-806141979_Spectrograms_20Hz.mat  181204-203002-437599-806141979
    Flow noise   20181209T083004-File-21Metadata   181209-083002-437599-806141979_Spectrograms_20Hz.mat  181209-083002-437599-806141979
    Fish         20181227T053004-File-10Metadata   181227-053002-437599-806141979_Spectrograms_20Hz.mat  181227-053002-437599-806141979
    Fish         20181227T100004-File-20Metadata   181227-100002-437599-806141979_Spectrograms_20Hz.mat  181227-100002-437599-806141979
    Whales       20181227T100004-File-6Metadata    181227-100002-437599-806141979_Spectrograms_20Hz.mat  181227-100002-437599-806141979
Interestingly the whale and second fish images come from the same file. We will import the 4 unique wav files from the external hard drive into our GitHub raw_data repo.

### Processing the data
1. Per the VGGish documentation,
"VGGish was trained with audio features computed as follows:

    All audio is resampled to 16 kHz mono.
    A spectrogram is computed using magnitudes of the Short-Time Fourier Transform with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann window.
    A mel spectrogram is computed by mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz.
    A stabilized log mel spectrogram is computed by applying log(mel-spectrum + 0.01) where the offset is used to avoid taking a logarithm of zero.
    These features are then framed into non-overlapping examples of 0.96 seconds, where each example covers 64 mel bands and 96 frames of 10 ms each."
To match those conditions, we will convert our wav files from their input form (mono, 48kHz, 32-bit) to the trained form (mono, 16kHz, 16-bit) using Audacity. Follow the steps in [this article](https://learn.adafruit.com/microcontroller-compatible-audio-file-conversion) to convert the audio files. Save the cleaned audio files in the intermediate_data directory. *NOTE: The lower the hertz the smaller the number of samples and the smaller your file will be when saved. BUT, you lose sound quality with lower sampling rates. Be careful about your choices. Upconverting a low bitrate (say 8 kHz to 16 to 22 kHz) will not make your file sound better.*
2. Copy the "poc/vggish/models/research/audioset/vggish/vggish_inference_demo.py" file and save your own version. Update the wav file path to your cleaned wav files in the "intermediate_data" directory.
3. Ensuring that your vggish_environment is activated, run the vggish_inference_demo.py code.

### Running the VGGish model on our data
1. In this POC we are interested in outputting embeddings for our 4 test files. We will not do any post-processing of data and will use raw_embeddings only. To that end Emily altered the vggish_inference_demo.py code to run only the 4 wav files, extract the raw embeddings, and save them in csv files in the embedding_data directory. If you'd like to look at the code, check out the scr/vggish_scripts. If you'd like to run the code do so in models/research/audioset/vggish.
2. If you're interested in collapsing the embeddings to 2 or 3 dimensions (originally 128) and visualizing the outputs please run the POC_PCA_graphing.ipynb in scr/data_processing.


## Data Files

### Raw Data Files
MLFigs_Labeled_Oct_26_Chris is a directory containing the json files which identify sounds in different spectrograms. We will use these json in "metadata_file_finder" to find the images which contain the most idenified sounds. 

\[]Metadata.txt - these files are the metadata files which correspond to the images found from the annotation jsons above.

### Intermediate Data
annotated_info.csv - This file contains the annotated sound labels, their coordinates, and the spectrogram and metadata file they belong to.

\[].wav - These are the preprocessed wav (audio) files of the images which had the most annotated sounds as found from the raw data above. For information on preprocessing please see the "Preprocessing the Data" section above. We only stored the preprocessed data (not the raw data) because GitHub didn't have room for both.

### Embedding Data Files
\[].csv - These csv files contain the embeddings output from the vggish model plus the numerical name of the recording file, what example number the embedding was for, start time in seconds of the example, and end time in seconds for the example.


## To Dos:

In Scope:
1. find wav files of interest
        - find the annotated images
        - see what the sounds are
        - choose a representative sample (or all) of them for analysis
        - see if we can match them back to .wav files
        - get those wav files in a directory we can use
2. base understand preprocessing code
3. run wav files through
4. get embeddings to run on one file
5. save embedding output
6. run embeddings on all files and save all outputs
7. run PCA or other dimensionality reduction to see clustering
8. graph PCA-reduced output

Out of scope:
1. Altering constants (e.g., hop length, windows)
2. labeling known sounds in embeddings
3. understanding post-processing


## Questions, Concerns, and Future Opportunitites

### Questions for Chris
1. Not all the annotated .png have metadata files. The two I found were 20190221T100004-File-13.png and 20190222T190004-File-28.png. They were both taken in 2019022. Have I misinterpreted how to tie the labeled spectrograms back to the .wav files which produced them? Additionally, I'm not hearing any clear sounds in the wav files which the labels would suggest they have.

### Concerns
1. We have a lot of "dead time" in our recordings (e.g., no anamolous sounds). This may muddy our results.
2. Still thinking about the training context of the model.

### Future Opportunitites
1. Resampling the audio from 16-bit to 32-bit and 48kHz to 16kHz causes us to lose a lot of depth of sound. It may be worth running the model on the original data or somewhat-resampled data to see how the groupings change.
2. VGGish currently cuts the wav files into ~1 second duration clips. How can we change these clips, windows, hops, etc. to alter the model, similar to the polar ice paper?
3. How does applying post-processing to our data impact the clusters of sounds?
4. Is PCA the best way to reduce the dimensonality for visualization? Is there another way which makes the clusters more apparent or captures more variation? (In POC we're capturing only 33% of variance in 2D and 46% in 3D).
5. Update code to remove depreciation warnings.


## Sources:
Where VGGIsh lives - https://github.com/tensorflow/models/tree/master/research/audioset/vggish
The collab with the how-to - https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF
Info on WAV files - https://en.wikipedia.org/wiki/WAV
TFR technical docs - https://www.thethingsnetwork.org/docs/devices/bytes/
TDS TFR article - https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
TDS explanation of log mel spectrogram - https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
PCA - https://medium.com/data-science-365/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0