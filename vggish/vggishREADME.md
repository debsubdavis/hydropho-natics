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
1. @Saumya - please add after finishing clustering module.


## Data Files Created

### Resampled .wav files

### Embedding CSV Files
\*.csv - These csv files contain the embeddings output from the VGGish model plus the numerical name of the recording file, what example number of the embedding, start time in seconds of the example, and end time in seconds for the example.