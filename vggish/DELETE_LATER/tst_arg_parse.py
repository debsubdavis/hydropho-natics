#testing functionality of arg parse

import argparse
import glob
import os

#argparse arguments
# Define argparse fields to allow for easier function querying
parser = argparse.ArgumentParser(description="Process wav files into log mel "+
                                 "spectrograms and feed into VGGish to create"+
                                 "embeddings")
parser.add_argument('--wav_path', action='store', required=True,
                    help="Relative path to wav files. "+
                    "Should contain signed 16-bit PCM samples.")
args = parser.parse_args()
#python vggish_audio_embeddings.py 
#--wav_path ../../../../../../../../../d/1Dec2018_28Feb2019/Hydrophone/



# Create a list of files from the input path
if args.wav_path:
    wav_path = args.wav_path
else:
    raise TypeError("Pass in relative path to wav files when calling the "+
                    "function e.g., \\"+
                    "$ python vggish_audio_embeddings.py "+
                    "--wav_path path/to/wav/files/'")
  
# Create the match pattern looking for wav files
'''if wav_path[-1] == '/':
    match_pattern = '/' + wav_path
elif wav_path[-1] != '/':
    match_pattern = '/' + wav_path'''
match_pattern = 'D:/1Dec2018_28Feb2019/Hydrophone/'
#'D:\\1Dec2018_28Feb2019\\Hydrophone\\'
print(type(match_pattern))
print(match_pattern)
#file_list = glob.glob('d/1Dec2018_28Feb2019/Hydrophone/*')
file_list = [file for file in os.listdir(match_pattern) if file.endswith('.wav')]
print(len(file_list))

#print(file_list)
#formatted_file_list = [file.replace('\\', '/') for file in file_list]
#print(len(formatted_file_list))