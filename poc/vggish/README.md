This README contains information to run the test VGGish implementation

## About
For the initial POC we wanted to test the clusters produced for existing known sounds in wav files. By comparing the inter-cluster and intra-cluster similarity we can get a sense of the goodness of clustering. We can also visually see how the clusters look together. Potential limitations of this method include: the potentially small number of observations available per sound-type, the reduction from 128 dimensions (number of dimensions created from the embedding) to 2 or 3 dimensions may not accurately reflect the distance of the clusters in space.


## Steps to Reproduce Analysis

### Setting up and Testing VGGish
1. Activate the vggish_environment virtual environment
2. Download the "requirements.txt" file from https://github.com/tensorflow/models/tree/master/research/audioset/vggish and move it into the directory you plan to run your code in. For us it's hydropho-natics/poc/vggish.
3. Run "pip install -r requirements.txt" which installs the dependent libraries. These libraries should be contained in the vggish_environemnt file, but this will cover you in the event that they are not.
4. *Taken from the TensorFlow VGGish README linked above*
    - Clone TensorFlow models repo into a 'models' directory.
    $ git clone https://github.com/tensorflow/models.git
    $ cd models/research/audioset/vggish
    - Download data files into same directory as code.
    $ curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
    $ curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
5. Execute the smoke test *Taken from the TensorFlow VGGish README linked above*
    - Installation ready, let's test it.
    $ python vggish_smoke_test.py
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


## Data Files

### Raw Data Files
### FINISH THIS SECTION

### Intermediate Data
annotated_info.csv - This file contains the annotated sound labels, their coordinates, and the spectrogram and metadata file they belong to.

## To Dos:
1. find wav files of interest (12/7)
    - find the annotated images
    - see what the sounds are
    - choose a representative sample (or all) of them for analysis
    - see if we can match them back to .wav files
    - get those wav files in a directory we can use
**2. understand preprocessing code (12/7-8)**
3. run wav files through (12/8)
4. get embeddings to run on one file (12/11-12)
**5. understand how to label embeddings
**6. save embedding output
**7. run embeddings on all files and save all labeled outputs
**8. run PCA or other dimensionality reduction to see clustering
**9. graph PCA-reduced output
**10. see if can tie back datapoints to their identified sounds

This is the new to-do list




## Concerns
1. VGGish will cut the wav files into ~1 second duration clips. How do we match the cuts back to the original WAV files and label the data?
2. What happens to the "dead time" in each file where there are no sounds?
3. Changing the file from mono 48kHz, 32-bit float to mono 16kHz 16-bit float appears to significantly reduce the amount of information available in the sound. Would model performance be degraded by leaving the files as-is?
4. There doesn't appear to be many sounds in the hydrophone clips. Are we sure these are the right files corresponding to the labeled datapoints?

## QUESTIONS FOR CHRIS
1. Not all the annotated .png have metadata files. The two I found were 20190221T100004-File-13.png has no metadata file and 20190222T190004-File-28.png has no metadata file. They were both taken in 2019022
2. What are our thoughts on resampling the audio from 16-bit to 32-bit and 48kHz to 16kHz? It felt like we lost a lot of the depth of sound, and maybe it would be worth seeing how the results of the model changed if we kept the file in its original format.


## Check out:
Where VGGIsh lives - https://github.com/tensorflow/models/tree/master/research/audioset/vggish
The collab with the how-to - https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF
Info on WAV files - https://en.wikipedia.org/wiki/WAV


