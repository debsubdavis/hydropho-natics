## About
This README contains information on the VGGish model and how to create audio embedddings for later clustering and identification.

## Goal
The goals of the codes in this folder are twofold. First, 

## Model Notes - VGGish
VGGish requires a log mel spectrogram represented as a 3-D matrix with dimensions [frames x 96 window length x 64 mel bins]. Don't worry about creating this input yourself, running the code as-is will produce the correct result. VGGish then returns an example (datapoint) representing 0.96 seconds of audio encoded in 128-D. Sometimes users will want to input spectrograms so that the output example is longer or shorter than 0.96 seconds. This can be achieved by changing the STFT_WINDOW_LENGTH_SECONDS, STFT_HOP_LENGTH_SECONDS, EXAMPLE_WINDOW_SECONDS, and EXAMPLE_HOP_SECONDS. Begin by changing the EXAMPLE_WINDOW_SECONDS and EXAMPLE_HOP_SECONDS to the values you desire in the vggish_params.py file ("hop" is the overlap between examples - our tests used 0 overlap, meaing that EXAMPLE_WINDOW_SECONDS = EXAMPLE_HOP_SECONDS). The EXAMPLE_WINDOW_SECONDS / STFT_HOP_LENTGH_SECONDS must equal 96. Alter the STFT_HOP_LENGTH_SECONDS to ensure EXAMPLE_WINDOW_SECONDS divided by it equals 96. STFT_HOP_LENGTH_SECONDS typically equals STFT_WINDOW_LENGTH_SECONDS * 0.4 (there's typically a 40% overlap when calculating the Fast Fourier Transform). By altering these 4 parameters you can continue to use the program as written while changing the amount of time represented in each example.

Formulaic value creation:
EXAMPLE_WINDOW_SECONDS: x
EXAMPLE_HOP_SECONDS: x (assuming 0 overlap between examples)
STFT_WINDOW_LENGTH_SECONDS: (x/96)/0.4
STFT_HOP_LENGTH_SECONDS: x/96

Inputs for 0.96 second embeddings, 40% overlap:
EXAMPLE_WINDOW_SECONDS: 0.96
EXAMPLE_HOP_SECONDS: 0.96
STFT_WINDOW_LENGTH_SECONDS: 0.025
STFT_HOP_LENGTH_SECONDS: 0.010

Inputs for 2 second embeddings:
EXAMPLE_WINDOW_SECONDS: 2
EXAMPLE_HOP_SECONDS: 2
STFT_WINDOW_LENGTH_SECONDS: 5/96
STFT_HOP_LENGTH_SECONDS: (1/48)

Inputs for 5 second embeddings:
EXAMPLE_WINDOW_SECONDS: 5
EXAMPLE_HOP_SECONDS: 5
STFT_WINDOW_LENGTH_SECONDS: 25/192
STFT_HOP_LENGTH_SECONDS: 5/96

Inputs for 20 second embeddings:
EXAMPLE_WINDOW_SECONDS:
EXAMPLE_HOP_SECONDS:
STFT_WINDOW_LENGTH_SECONDS:
STFT_HOP_LENGTH_SECONDS:


## Model Notes - Clustering


## Steps to Execute Analysis

### I. Set up environment and verify VGGish runs via a "smoke test"
1. From the vggish directory, create and activate the vggish virtual environment using the below code:
    - $ conda env create -f environment.yml 
    - $ conda activate vggish_run_environment
2. Navigate to vggish/scr and run the following code to install dependent libraries. They should already be included in the Vggish environment.yml file, but this will cover you in the event that they are not.
    - $ pip install -r requirements.txt
3. Vggish also requires downloading two data files. Download the below files and move them into the vggish/scr directory.
    - [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt),
    in TensorFlow checkpoint format.
    - [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz),
    in NumPy compressed archive format.
4. Execute the smoke test script from the vggish/scr directory using the commands below. You will need to uncomment the import lines in the vggish_input.py file and comment the ones starting with "from .". Switching the imports is done for testing purposes on GitHub.
    - $ python vggish_smoke_tst.py
    - If we see "Looks Good To Me", then we're all set.
    - If you don't get the "Looks Good to Me" message, interpret the error code and/or try following the steps from the VGGish [README](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md). A common reason you may get the "tensorflow has no python module" error is because you aren't in the vggish_environment (or didn't pip install the requirements.txt).

### II. Getting WAV files for analysis
1. Create a directory containing all your .wav files for analysis.
2. The code assumes that all audio is sampled as 16kHz mono. If your data doesn't come in this format, please run the audio_resampling.py code in the scr directory. You will need to input the path to your wav files and the save path for resampled audio. NOTE: The audio_resampling code will create new copies of the .wav files. Verify you have sufficient storage space before running it.

### III. Creating Audio Embeddings
Per the TensorFlow VGGish documentation:
"VGGish was trained with audio features computed as follows:
    *All audio is resampled to 16 kHz mono.
    *A spectrogram is computed using magnitudes of the Short-Time Fourier Transform with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann window.
    *A mel spectrogram is computed by mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz.
    *A stabilized log mel spectrogram is computed by applying log(mel-spectrum + 0.01) where the offset is used to avoid taking a logarithm of zero.
    *These features are then framed into non-overlapping examples of 0.96 seconds, where each example covers 64 mel bands and 96 frames of 10 ms each."
In development for the 2024 MSDS Capstone project, the team left all model parameters unchanged (e.g., hop length, window length, example length) given the higher likelihood of good model performance if training and testing conditions matched.
1. Ensuring that your vggish_environment is activated, run the vggish_inference_demo.py code as below:
      $ python vggish_audio_embeddings.py --wav_path path/to/wav/files/ --save_path path/to/where/embeddings/should/go/
Audio embeddings will be saved as csv files (1 csv per wav file). No post-processing will be done on the data.
You will note several depreciation warnings. VGGish was created to be compatible with TensorFlow 1, not TensorFlow 2 (most current version). The TensorFLow version has been fixed at 2.15 in the requirements.txt document as well as in the environment.yml.

### IV. Performing dimensionality reduction and clustering
1. @Saumya - please add after finishing clustering module.


## Data Files

### Embedding Data Files
\*.csv - These csv files contain the embeddings output from the vggish model plus the numerical name of the recording file, what example number of the embedding, start time in seconds of the example, and end time in seconds for the example.

### Questions for Chris
1. Validate that downsampling, 16-bit PCM changes methodology in the code are correct (code uses ‘soxr_vhq’ resampling - Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation. 'soxr_hq' is the default setting of soxr.)