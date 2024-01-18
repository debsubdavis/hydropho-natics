# DELETE - DEVELOPER NOTES
*Pull requests submitted to make changes to the main branch should come from the dev-vgg branch.
*Please list all needed libraries in the requirements.txt file in this directory
    - Will add libraries to the environment.yml file in the highest level of the hydrophanatics directory as needed. Please do not make changes directly to the environment.yml file at the highest level of the directory
*If you would like to create an environment.yml for local development purposes, please create one in the vggish directory. Recommend changing the "name" field within the environment.yml to something besides hydrophonatics

## About
This README contains information to create audio embedddings via the TensorFlow VGGish model.

### INSERT MORE ON WHAT THE GOAL IS, THE MODEL, WHERE ITS FROM, WHAT ALL THE CODE IN THIS DIRECTORY DOES INCLUDING TSNE

## Steps to Execute Analysis

### Set up environment and verify VGGish runs via a "smoke test"
1. Activate the vggish_environment virtual environment. The information is stored in the vggish_environment.yml.
2. Download the "requirements.txt" file from https://github.com/tensorflow/models/tree/master/research/audioset/vggish and move it into the directory you plan to run your code in. All the code is stored in the vggish/code/ directory.
3. Run "pip install -r requirements.txt" which installs the dependent libraries. These libraries should be contained in the vggish_environemnt file, but this will cover you in the event that they are not.
4. Execute the smoke test *Taken from the TensorFlow VGGish README linked above*
    - $ python vggish_smoke_test.py
    - If we see "Looks Good To Me", then we're all set.
    - If you don't get the "Looks Good to Me" message, interpret the error code and/or try following the steps from the Vggish readme linked above. A common reason you may get a error which looks like "tensorflow has no python module" is because you aren't in the vggish_environment (or didn't download the requirements.txt).

### Getting WAV files for analysis
1. Create a directory containing all your .wav files for analysis. The code assumes that all code is sampled as 16kHz mono. If your data doesn't come in this format originally please resample using Audacity or Python libraries.

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
1. In this POC we are interested in outputting embeddings for our 4 test files. We will not do any post-processing of data and will use raw_embeddings only. To that end Emily altered the vggish_inference_demo.py code to run the wav files, extract the raw embeddings, and save them in csv files in the embedding_data directory. If you'd like to look at the code, check out the scr/vggish_scripts. If you'd like to run the code do so in models/research/audioset/vggish.
2. If you're interested in collapsing the embeddings to 2 or 3 dimensions (originally 128) and visualizing the outputs please run the POC_PCA_graphing.ipynb in scr/data_processing.


## Data Files

### Embedding Data Files
\[].csv - These csv files contain the embeddings output from the vggish model plus the numerical name of the recording file, what example number the embedding was for, start time in seconds of the example, and end time in seconds for the example.


## To Dos:

In Scope:

Nice to have:
0. Optional resampling for wav files in python (so user doesn't have to have pre-pre-processed or run in Audacity)
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
TFR technical docs (file structure) - https://www.thethingsnetwork.org/docs/devices/bytes/
Towards Data Science TFR article - https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
Towards Data Science explanation of log mel spectrogram - https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
PCA - https://medium.com/data-science-365/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0