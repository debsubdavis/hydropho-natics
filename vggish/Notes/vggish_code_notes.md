## Purpose
The purpose of this document is to house the notes on how VGGish tactically functions. The goals are 2-fold:
    1. Modelers will understand in depth how the model works.
    2. Modelers can identify assumptions/constants/areas of opportunity/areas of tweaking.

## Approach
To ensure that Emily and Saumya can understand the model within the 15 hrs/week alotted, they will each understand different "chunks" of the model (represented by files, functions, etc.), then come together to explain how they work to the other. The elements of vggish code are as follows (taken from the Tensor Flow  readme markdown in vggish/models/research/audioset/vggish)
    * `vggish_slim.py`: Model definition in TensorFlow Slim notation.
    * `vggish_params.py`: Hyperparameters.
    * `vggish_input.py`: Converter from audio waveform into input examples.
    * `mel_features.py`: Audio feature extraction helpers.
    * `vggish_postprocess.py`: Embedding postprocessing.
    * `vggish_inference_demo.py`: Demo of VGGish in inference mode.
    * `vggish_train_demo.py`: Demo of VGGish in training mode.
    * `vggish_smoke_test.py`: Simple test of a VGGish installation

*SEE ASSUMPTIONS & NEXT STEPS SECTION BELOW FOR WHY CODES WERE REMOVED FROM LIST REQUIRING DEEP UNDERSTANDING*

Codes which are out of scope for deep understanding:
    * `vggish_train_demo.py`: Demo of VGGish in training mode.
    * `vggish_smoke_test.py`: Simple test of a VGGish installation
    * `vggish_postprocess.py`: Embedding postprocessing.

Codes which are in scope for deep understanding:
    * (All) `vggish_inference_demo.py`: Demo of VGGish in inference mode.
    * (EKRC) `vggish_input.py`: Converter from audio waveform into input examples.
        * `vggish_params.py`: Hyperparameters.
        * `mel_features.py`: Audio feature extraction helpers.
    * (SN) `vggish_slim.py`: Model definition in TensorFlow Slim notation.
        * `vggish_params.py`: Hyperparameters.

Given we will likely be running our model in a method similar to vggish_inference_demo and it combines all the modeling codes, it will be important for both teammates to understand. 

## Assumptions
### Code out of scope for deeper understanding
vggish_train_demo.py - *Emily's opinion* We've historically assumed that retraining the model is out of scope, thus vggish_train_demo could be removed from the list of codes requiring deeper understanding. However, it may be beneficial to circle back to this code if we have time to train vggish on our use cases.

vggish_smoke_test.py - This is to test the quality of the install, not for actual operation.

vggish_postprocess.py - Applying post-processing to our raw embeddings is currently listed as an area of expansion. We will hold off on understanding this code in depth for now.

## Areas for future investment
*Understand vggish_postprocess.py to see if postprocessing our data creates better/different clusters
*Understand the vggish_train_demo code to see if retraining makes sense

## Code Notes
### vggish_inference_demo


### vggish_input - Emily Creeden
Background:
We know from the vggish_inference_demo "A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted into log mel spectrogram examples..."
We also know "The input size was changed to 96x64 for log mel spectrogram audio inputs." 
From the README:
VGGish was trained with audio features computed as follows:

* All audio is resampled to 16 kHz mono. (see questions for chris)
# What does the below mean?
* A spectrogram is computed using magnitudes of the Short-Time Fourier Transform
  with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann
  window.
    * Short-Time Fourier Transform - decomposes signal into individual sin waves which correspond to different frequencies. It moves the audio from the time dimension (e.g., all the sins on top of each other) to the frequency dimension (peaks for each frequency in the recording). A short-time fourier transform is the fast fourier transform computed on overlapping windowed segments of the signal because the sound isn't periodic (repeating).
    * window size - "time chunks" of the input signal (e.g., break the signal into x smaller chunks by time)
    * window hop - if the windows overlap, this is the distance between the start of the window and the overlapping section. From stackoverflow - "Hop size should refer to the number of samples in between successive frames (windows)"
    * periodic Hann window - " the Hanning window smoothly tapers the endpoints to zero and mitigates the discontinuity that produces leakage." [Science Direct](https://www.sciencedirect.com/topics/engineering/hanning-window). Per Texas Instruments "For the FFT, both the time domain and the frequency domain are circular topologies, so the two endpoints of the time waveform are interpreted as though they were connected together. When the measured signal is periodic and an integer number of periods fill the acquisition time interval, the FFT turns out fine as it matches this assumption. [...]
    When the number of periods in the acquisition is not an integer, the endpoints are discontinuous. These artificial discontinuities show up in the FFT as high-frequency components not present in the original signal. The spectrum you get by using a FFT, therefore, is not the actual spectrum of the original signal, but a smeared version. It appears as if energy at one frequency leaks into other frequencies. This phenomenon is known as spectral leakage, which causes the fine spectral lines to spread into wider signals. [...]
    You can minimize the effects of performing an FFT over a noninteger number of cycles by using a technique called windowing. Windowing reduces the amplitude of the discontinuities at the boundaries of each finite sequence acquired by the digitizer. Windowing consists of multiplying the time record by a finite-length window with an amplitude that varies smoothly and gradually toward zero at the edges. This makes the endpoints of the waveform meet and, therefore, results in a continuous waveform without sharp transitions. This technique is also referred to as applying a window. [...]
    In general, the Hanning (Hann) window is satisfactory in 95 percent of cases. It has good frequency resolution and reduced spectral leakage. If you do not know the nature of the signal but you want to apply a smoothing window, start with the Hann window.". Per Math Works help center "Periodic is useful for spectral analysis because it enables a windowed signal to have the perfect periodic extension implicit in the discrete Fourier transform. When 'periodic' is specified, hann computes a window of length L + 1 and returns the first L points."
    * What did chris use in his paper? - They appear to use a variety of windows. They are looking a power spectral density. They chose 1 min windows to identify flow noise. They also had "PSDs for spectrograms used in ML are processed using 0.1-second windows, tapered with a Hann window, and 50% overlap. [...] Processed spectrograms for ML were one minute in duration with frequency limits from 0-1 kHz." and "For the spectral fitting algorithms PSDs are calculated using using 1-second windows, tapered with a Hann window, and 50% overlap." Later in the paper they say "Lastly, each recording was approximately 30-minutes long but processed in one-minute windows."
    * What did they use in the ice paper? - They used 2, 5, and 20 second windows with 50%, 90% and 90% overlap respectively. They found in their model (though they were trying to identify noises AS WELL AS when they started and stopped) "the 20 second window has a very coarse temporal resolution while the 2 and 5 second windows perform relatively well and have good temporal resolution. One might also note that the 2 second window predictions are more sporadic and noisy relative to the 5 and 20 second window predictions, which is likely due to the fact that the 2 second window has the least context available to make predictions."
* A mel spectrogram is computed by mapping the spectrogram to 64 mel bins
  covering the range 125-7500 Hz.
    * Mel spectrogram - Per the Towards Data Science article "Studies have shown that humans do not perceive frequencies on a linear scale. We are better at detecting differences in lower frequencies than higher frequencies. For example, we can easily tell the difference between 500 and 1000 Hz, but we will hardly be able to tell a difference between 10,000 and 10,500 Hz, even though the distance between the two pairs are the same. In 1937, Stevens, Volkmann, and Newmann proposed a unit of pitch such that equal distances in pitch sounded equally distant to the listener. This is called the mel scale. We perform a mathematical operation on frequencies to convert them to the mel scale."
    * 64 mel bins? - Per the Getting to Know Mel Spectrograms page, the Mel binning transforms the "Hz scale into bins, and transforms each bin into a corresponding bin in the Mel Scale". The frequencies in the range 125-7500Hz are what's binned.
* A stabilized log mel spectrogram is computed by applying
  log(mel-spectrum + 0.01) where the offset is used to avoid taking a logarithm
  of zero.
    * Stabilized log mel spectrogram? - When you take the log of the mel spectrogram value you add a small amount to avoid taking the log of 0 (which could happen if the mel spectrogram is equal to 0 after binning)
* These features are then framed into non-overlapping examples of 0.96 seconds,
  where each example covers 64 mel bands and 96 frames of 10 ms each.
    * what are features? - features are generally used to train a model. I would guess that each stabilized log mel spectrogram is a feature.
    * How does the non-overlapping examples of 0.96 seconds relate to the original windows/hops? 
    * are mel bands and mel bins the same?
    * how do we get from 96 frames of 10ms each back to 0.96 seconds?


#### Questions for Chris
1. Help me understand why the output FFT don't overlap when the windows overlap (see diagram from the TDS understanding the mel spectrogram article)? 
As I take it: ideally we would run the FFT on the whole sound. Unfortunately though, the entire sound isn't periodic so the FFT will "smudge" the sound because it has discontinuous ends. Instead we use STFT on smaller overlapping windows (corrected with the Hann window to produce continuous window ends). When we run FFT on these windowed segments it somehow produces non-overlapping FFTs which show the amplitudes of each frequency in chunks of the audio duration, which if summed together would represent the whole audio. To show the amplitude of all the frequencies in a whole audio one would superimpose (or add) all the amplitudes from the time chunked audio together. Is that correct? Why bother creating overlapping windows in the first place and not just run the whole audio file through the FFT to produce a single graph? If the windows you run FFT on overlap, why don't the FFT overlap (per this diagram which I'm putting lots of stock into)
2. I resampled our audio to 16kHz mono and appeared to lose a lot of information looking at the waveform. Is there a better way to do this than in Audacity?
3. For the YOLO model it appears each 30 min recording is broken down into 1 min windows. Those 1 min windows are then broken down into 1 second or 0.1 second windows with Hann windows and 50% overlap in both cases. We currently are processing the whole signal (can be 30 min) into 25ms windows with 40% (10ms) overlap and a periodic Hann window. Do we think any tweaks should be made there?
4. The model uses a mel spectrogram with 64 mel bins and frequencies in the range 125-7500 Hz. Does that frequency range feel like it captures the entirety or the valuable part of what we might see? Any thoughts on the binning?

#### Questions to think through w/ Saumya
#### References
Texas Instruments Signals info - https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf

Towards Data Science Understanding the Mel Spectrogram - https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

Towards Data Science Getting to Know the Mel Spectrogram - https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

Towards Data Science YAMNet explanation - https://farmaker47.medium.com/classification-of-sounds-using-android-mobile-phone-and-the-yamnet-ml-model-539bc199540




### vggish_slim - Saumya Nauni
#### Questions for Chris
#### Questions to think through w/ Emily
