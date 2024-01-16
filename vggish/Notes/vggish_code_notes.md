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


## Code Notes
### vggish_input Notes - Emily Creeden
Background:
We know from the vggish_inference_demo "A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted into log mel spectrogram examples..."
We also know "The input size was changed to 96x64 for log mel spectrogram audio inputs." 
From the README:
VGGish was trained with audio features computed as follows:
* All audio is resampled to 16 kHz mono. (see questions for chris)
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
    * How does the non-overlapping examples of 0.96 seconds relate to the original windows/hops? I think this is created by how the FFT are combined from the overlapping windows to result into 0.01 second frames combined into 0.96 second examples. So basically we break down the total audio into the windows with overlap, compute the FFT on the windows, then recombine the pieces into 0.01 second bits, combined into 0.96 second chunks.
    * are mel bands and mel bins the same? - It seems so (see On Mel Bands below). Frequencies are binned into Mel bands. The code in _params also has NUM_BANDS = NUM_MEL_BINS

### vggish_input Notes - Emily Creeden
* seems like can change the sample rate both for the input file and the model, though it was trained with specific parameters - worth double checking that "sr" is what we understand based on what we have in our input file
* we can edut the upper and lower frequency edges
* the "examples" and "features" are different. The features are the log mel spectrograms with their own window length and hop length in seconds for calculation purposes.
* Examples have their own window lengths and hop lengths, but the Example's parameters aren't in seconds, they're in features. The example_window (s) * feature sample rate (feature/s) creates the example window length. In our demo case,
1/0.010s, or 100 features/second, or 100 log mel spectrograms per second of audio.
* We get an example window length of 96 features (log mel spectrograms representing 0.010 seconds of audio)
* We get an example hop length of 96 features as well, meaning there is no overlap in Examples

* *SEE mel_features below*
* Once we have all the mel features, they get framed into the examples
* The example window length is in features (96)
* The example hop length is also in features (96). Because it's the same length as the window, there is no overlap (but we could overlap them if we wanted to)
* Return to mel_features.frame()
    * input is log_mel spectrogram matrix (frames (same as original spectrogram framing) x 64), window_length = 96 features, and hop length = 96 features
    * We input a 2-d array, so the output will be a 3-d array (num_frames, window_length, ...)
    * This is where the regrouping ("reframing") into examples 0.96s in length consisting of 96 frames of 0.010 occurrs



### mel_features Notes - Emily Creeden
* log_mel_spectrogram -> stft_magnitude -> frame -> periodic_hann -> spectrogram_to_mel_matrix -> hertz_to_mel
* log_mel_spectrogram - converts waveform into log mag mel-freq spectrogram
    * takes in your data (wav file resampled to rate assumed by VGGish - 16kHz), then the _params for the rest of the fields to create the log mel spectrogram - there are saved values in case you don't put a field in
    * the bulk of this code seems to put the window and hop length into samples/events instead of seconds and generates a fft length in samples/events
    * *SEE stft_magnitude*
    * *SEE spectorgram_to_mel_matrix*
* stft_magnitude - calculates stft magnitude
    * now calls the data (wav file resampled to params sample rate) "signal"
    * returns the stft magnitude for each frame of input samples
    * *SEE frame below*
    * now that data is framed, a periodic hann is applied
    * *SEE periodic_hann below*
    * creates windowed frames a.k.a., the data divided into frames (based on the window length and overlap) and multiplied by the Hann window - this seems like it creates the "pinched audio wave samples" from the TDS article on Mel spectrograms
    * then the numpy function fft.rfft takes the 1-d discrete FT on the input
    * IMPORTANT - per the fft.rfft documentation  the second input in the function is the "number of points along transformation axis in the input to use. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If n is not given, the length of the input along the axis specified by axis is used." In our case n = int(fft_length) = 512 which is longer than the 400-sample length windows, so our input is padded with zeros
    * It returns only the non-neg frequency terms w/ the function and the abs (fft_length/2 + 1) - so for every frame, it returns the magnitudes of the frequencies from the events in the window as the spectrogram
* frame - converts array into sequence of successive possibly overlapping frames
    * breaks the big n-dimensional array of samples (signal now called data again) into a 2-d array which has rows = # of frames (calculated in code), and each row has window_length # of samples
    * it uses stride_tricks (probably a library) to avoid copying the data again
    * it returns 2-D array (because input is always 1D) which has # of rows = complete frames available in the input data (contents of the rows are window length # of events/samples). I'm imagining the following:
        input array: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        window length = 4, overlap = 2
        output array with window length and overlap above:
        [[1,2,3,4],
        [3,4,5,6],
        [5,6,7,8],
        [7,8,9,10],
        [9,10,11,12],
        [11,12,13,14]] <- 15 was cut off because the next row doesn't have enough #
* periodic_hann - calculates a periodic Hann window
    * creates an array for calculating the periodic Hann window
* spectrogram_to_mel_matrix - 
    * num_spectrogram_bins = shape of the second dimension of the spectrogram (mangitudes of the fft_length/2+1 (257) unique values of the fft per frame), audio_sample_rate is still 16kHz
    * It looks like we pass in the rest of the **kwargs from log_mel_spectrogram into spectrogram_to_mel_matrix (so the preset values aren't used)
    * The function multiplies the stft_magnitudes matrix by a matrix which converts it into a mel spectrogram. The mel spectrogram is represented in a 2-d array with num_spectrogram_bins rows (257) and num_mel_bins columns (64)
    * When we multiply the spectrogram (frames x 257) and spectrogram_to_mel_matrix (257x64) together, we get a resulting mel spectrogram matrix of (frames x 64)
    * No time changes (e.g., from 0.025 second stft window to final 0.96 second examples made of 96 10ms frames).
* hertz_to_mel - switches frequencies from hertz to mel

## Questions for Chris
1. Our model first creates log mel spectrograms from 0.025s windows with 0.010 overlap. These log mel spectrograms are then combined to represent 0.96 seconds of total audio with no overlap. A key factor in that math appears to be 1/hop length of those original log mel spectrogram windows. I had been thinking of this as the "unique sound contribution" of each spectrogram. Is there a better way to think about it? Basically yes.

2. When we spoke last time you had some thoughts on the windowing/overlapping of the sample. I noted that for the YOLO model it appears each 30 min recording is broken down into 1 min windows. Those 1 min windows are then broken down into 1 second or 0.1 second windows with Hann windows and 50% overlap in both cases. We currently are processing the whole signal into 0.96 second windows (consistent with the model training) made up of 25ms windows with 40% (10ms) overlap and a periodic Hann window. I'd recommend keeping the data processed as the model was originally trained. If we have time, planning to play with the windows to see if we get better results. Any thoughts on how to go about this or pre-existing windows that you favor? Hold for now. The hop length and total length will impact the time resolution and frequency resolution of the systems. in a longer window length will give better frequency resoluion and poorer time resolution - can make alterations to get better resolutions. We're trading time for frequency but they don't overlap at all.

3. The examples timing (1 second vs. longer?) - seems like something we can play around with. 1 second is pretty good balance of time and spectral resolution. Thinks we may want to pay the price in time resolution - don't need to have the window lengths as short as they are b.c we aren't gaining anything from it.

4. I resampled our audio to 16kHz mono and appeared to lose a lot of information looking at the waveform. Is there a better way to do this than in Audacity? - the signal gets much quieter because you cut out the popcorn shrimp, it will have a different impact on different files. Don't worry about it. Can downsample in python. There are subtleties but they don't have anything to do with you. DO IT IN PYTHON FOR EASE OF REUSE. The downsampling will remove the high frequency noises. Chris would band-pass filter it but it won't impact us. The things that we care about happen at lower frequencies. It hits the high frequencies harder because they are shorter so will get frewer of the points randomly sampled in the downsampling.

5. The model uses a mel spectrogram with 64 mel bins and frequencies in the range 125-7500 Hz. Does that frequency range feel like it captures the entirety or the valuable part of what we might see? Any thoughts on the binning? Chris would like it to be higher ideally, but dont let it be a deal breaker. Until recently no one had shown high freqiency noises from a WEC but when Chris went into the WEC there was a lot of high frequency noises. Tidal was 8kHz and 16kHz from variable drive keeping the DC motor functioning. They are modulating the current through the motor- so there is an electrical switch and the components which go through the motors vibrate at high frequencies - happens at the drive frequencies. On the low frequency end flow noise gets in the way - doesn't care. let's explore methods.


## Questions for/to think through w/ Saumya
1. What is the mel-spectrogram patch? Is that part of the model?


## Internal Questions
1. Does our sample lie in the range -1 to 1 as the model expects? Seems like it must based on wavfile_to_examples / 32768 (magic number?)
2. Should the audio_sample_rate in log_mel_spectrogram be 16khz instead of 8kHz as it's initially hard coded?


## Areas for future investment
* Changing the parameters of the mel spectrogram creation or example length to better capture sounds
* Understand vggish_postprocess.py to see if postprocessing our data creates better/different clusters
* Understand the vggish_train_demo code to see if retraining makes sense


## References
* Texas Instruments Signals info - https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
* Towards Data Science Understanding the Mel Spectrogram - https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
* Towards Data Science Getting to Know the Mel Spectrogram - https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
* Towards Data Science YAMNet explanation - https://farmaker47.medium.com/classification-of-sounds-using-android-mobile-phone-and-the-yamnet-ml-model-539bc199540
* On Mel Bands - https://learn.flucoma.org/reference/melbands/


## Reading list
* VGG architecture - https://paperswithcode.com/method/vgg 
* Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks - https://arxiv.org/pdf/1706.07156.pdf 
* What are frequency and temporal resolution? - (https://www.avisoft.com/Help/SASLab/menu_main_analyze_spectrogram_parameters.htm#:~:text=Resolution%20The%20frequency%20resolution%20depends,sample%20rate%20%2F%20FFT%20length).)