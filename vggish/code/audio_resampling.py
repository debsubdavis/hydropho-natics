'''The purpose of this file is to resample audio files in a programmatic way
The audio needs to have the following qualifications:
    1. Signed 16-bit PCM samples
    2. Sampling rate of 16kHz
    3. Mono input
'''

import librosa
import soundfile as sf
import numpy as np
import argparse
import os


# Define argparse parameters to run from the command line
parser = argparse.ArgumentParser(description="Resample audio files for input "+
                                 "into VGGish model.")
parser.add_argument('--wav_path', action='store', required=True,
                    help="Path to directory containing .wav files.")
parser.add_argument('--target_sample_rate', action='store', required=False,
                    default= 16000,
                    help="Desired new sample rate of the audio. "+
                    "VGGish requires 16kHz.")
parser.add_argument('--save_path', action='store', required=True,
                    help='Path to directory where resampled wav files '+
                    'will be stored.')
args = parser.parse_args()
#python audio_resampling.py --wav_path D:/1Dec2018_28Feb2019/Hydrophone/ --save_path D:/1Dec2018_28Feb2019/Hydrophone_Resampled/

def main():
    #Resampling audio
    if args.wav_path:
        wav_path = args.wav_path
        save_path = args.save_path
        target_sample_rate = args.target_sample_rate

    #Get a list of the wav files
    file_list = [wav_path+file for file in os.listdir(wav_path) if file.endswith('.wav')]
    for wav_file in file_list:
        wav_data, sr = librosa.load(wav_file, sr=None, mono=True)
        raw_filename = wav_file[len(wav_path):-4]
        wav_filename = wav_file[:-4]
        save_filename = save_path + raw_filename + '_resampled.wav'
        # Resample and write out audio as 16bit PCM WAV
        wav_data_resampled = librosa.resample(y=wav_data, orig_sr=sr, target_sr=target_sample_rate)
        sf.write(save_filename, wav_data_resampled,
                target_sample_rate, format='wav', subtype='PCM_16')
        print("Processed {0}".format(raw_filename))

        #Uncomment the following code to verify that your file was changed into the format your expect
        #info = sf.info(save_filename)
        #print(info)

if __name__ == "__main__":
    main()