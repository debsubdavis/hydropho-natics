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

Codes which are in scope for deep understanding:
    * `vggish_inference_demo.py`: Demo of VGGish in inference mode.
    * `vggish_params.py`: Hyperparameters.
    * `vggish_slim.py`: Model definition in TensorFlow Slim notation.
    * `vggish_input.py`: Converter from audio waveform into input examples.
    * `mel_features.py`: Audio feature extraction helpers.
    * `vggish_postprocess.py`: Embedding postprocessing.


## Assumptions
### Code out of scope for deeper understanding
vggish_train_demo.py - *Emily's opinion* We've historically assumed that retraining the model is out of scope, thus vggish_train_demo could be removed from the list of codes requiring deeper understanding. However, it may be beneficial to circle back to this code if we have time to train vggish on our use cases.

vggish_smoke_test.py - This is to test the quality of the install, not for actual operation.



## Areas of Opportunity
*Understand the vggish_train_demo code to see if retraining makes sense