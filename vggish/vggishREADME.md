# DELETE - DEVELOPER NOTES
*Pull requests submitted to make changes to the main branch should come from the dev-vgg branch.
*Please list all needed libraries in the requirements.txt file in this directory
    - Will add libraries to the environment.yml file in the highest level of the hydrophanatics directory as needed. Please do not make changes directly to the environment.yml file at the highest level of the directory
*If you would like to create an environment.yml for local development purposes, please create one in the vggish directory. Recommend changing the "name" field within the environment.yml to something besides hydrophonatics

## About
This README contains information to create audio embedddings via the TensorFlow VGGish model.

### INSERT MORE ON WHAT THE GOAL IS, THE MODEL, WHERE ITS FROM, WHAT ALL THE CODE IN THIS DIRECTORY DOES INCLUDING TSNE

## Steps to Execute Analysis

### I. Set up environment and verify VGGish runs via a "smoke test"
1. Activate the vggish_environment virtual environment. The information is stored in the vggish_environment.yml.
2. Download the "requirements.txt" file from https://github.com/tensorflow/models/tree/master/research/audioset/vggish and move it into the directory you plan to run your code in. All the code is stored in the vggish/code/ directory.
3. Run "pip install -r requirements.txt" which installs the dependent libraries. These libraries should be contained in the vggish_environemnt file, but this will cover you in the event that they are not.
4. Execute the smoke test *Taken from the TensorFlow VGGish README linked above*
    - $ python vggish_smoke_test.py
    - If we see "Looks Good To Me", then we're all set.
    - If you don't get the "Looks Good to Me" message, interpret the error code and/or try following the steps from the Vggish readme linked above. A common reason you may get a error which looks like "tensorflow has no python module" is because you aren't in the vggish_environment (or didn't download the requirements.txt).

### II. Getting WAV files for analysis
1. Create a directory containing all your .wav files for analysis. The code assumes that all code is sampled as 16kHz mono. If your data doesn't come in this format originally please resample using Audacity or Python libraries. Follow the steps in [this article](https://learn.adafruit.com/microcontroller-compatible-audio-file-conversion) to convert the audio files in Audacity. *NOTE: The lower the hertz, the smaller the number of samples and the smaller your file will be when saved. BUT, you lose sound quality with lower sampling rates. Upconverting a low bitrate (say 8 kHz to 16 to 22 kHz) will not make your file sound better.*

### III. Creating Audio Embeddings
1. Per the VGGish documentation,
"VGGish was trained with audio features computed as follows:
    *All audio is resampled to 16 kHz mono.
    *A spectrogram is computed using magnitudes of the Short-Time Fourier Transform with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann window.
    *A mel spectrogram is computed by mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz.
    *A stabilized log mel spectrogram is computed by applying log(mel-spectrum + 0.01) where the offset is used to avoid taking a logarithm of zero.
    *These features are then framed into non-overlapping examples of 0.96 seconds, where each example covers 64 mel bands and 96 frames of 10 ms each."
#### SEE IF NEEDS TO BE CHANGED - The version of this model developed for the 2024 MSDS Capstone project will leave all model parameters as-is (e.g., hop length, window length, example length) given the higher likelihood of good model performance if training and testing conditions match.
2. Ensuring that your vggish_environment is activated, run the vggish_inference_demo.py code as below:
      $ python vggish_audio_embeddings.py --wav_path path/to/wav/files/
   #### By including '--wav_path path/to/wav/files/' you can avoid hard-coding or changing any parameters in the code itself. Audio embeddings will be saved as csv files (1 per input wav file) in a directory ("embedding_data/") on the same level as the directory where you run your code. No post-processing will be done on the data

### IV. Performing dimensionality reduction and clustering
1. XX


## Data Files

### Embedding Data Files
\[].csv - These csv files contain the embeddings output from the vggish model plus the numerical name of the recording file, what example number the embedding was for, start time in seconds of the example, and end time in seconds for the example.





## Questions, Concerns, and Future Opportunitites

### Questions for Chris
1. Not all the annotated .png have metadata files. The two I found were 20190221T100004-File-13.png and 20190222T190004-File-28.png. They were both taken in 2019022. Have I misinterpreted how to tie the labeled spectrograms back to the .wav files which produced them? Additionally, I'm not hearing any clear sounds in the wav files which the labels would suggest they have.

### Concerns
1. 

### To Dos:
1. update code to remove depreciation warnings
2. "bing" sound when code quits running