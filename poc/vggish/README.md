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
1. Copy the JSON annotation files from the Externa hard drive/Annotaion Stuff/MLFigs_Labeled_Oct_26 into raw_data/. We will use these annotated files for our intial POC.
2. Run the wav_file_finder to get the list of wav files for the POC.


## Data Files

### Raw Data Files
### FINISH THIS SECTION

### Intermediate Data
annotated_info.csv - This file contains the annotated sound labels, their coordinates, and the spectrogram they belong to.

## To Dos:
1. find wav files of interest (12/7)
    - find the annotated images
    - see what the sounds are
    - choose a representative sample (or all) of them for analysis
    - see if we can match them back to .wav files
    - get those wav files in a directory we can use
2. understand preprocessing code (12/7-8)
3. run wav files through (12/8)
4. get embeddings to run (12/11-12)
5. create initial clusters of data (12/12-13)

## Concerns
1. Multiple sound are labeled in each image? How do we attribute the sounds to particular cuts of the wav files?

## Check out:
Where VGGIsh lives - https://github.com/tensorflow/models/tree/master/research/audioset/vggish
The collab with the how-to - https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF

