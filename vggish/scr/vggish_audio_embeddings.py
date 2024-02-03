# This code leverages code written by TensorFlow. The TensorFlow
# code uses the following license and information:
#
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Code used to extract audio embeddings from wav files using TensorFlow's
VGGish model.

This code will leverage functions written by TensorFlow including
vggish_input.py and vggish_params.py to turn the wav files in the input path
into log mel spectrograms, feed them into VGGish, and generate 128-D raw
embeddings. Currently no post-processing is applied to the data, though this
feature may be added in the future. If the embedding code is run multiple times
with the same file names, the output csvs with the embeddings will be
overwritten. If you plan to re-run the code on files multiple times, rename
the resulting csvs or move them to avoid overwriting the outputs.

Input(s):
  # Path to directory containing wav files for analysis. Users should
  use '/' or '\\' to separate levels of directories.
  # Path to directory where embeddings should be stored

Output(s):
  # One csv file per input wav file which contains the 128 embeddings plus
  original recording info (name of wavfile, example number, start time of 
  example in seconds from original wav file, and stop time of example in
  seconds from original wav file. This file is stored in a directory called
  "embedding_data" which is one level above the current directory

Usage:
  # Run a WAV file through the model and save the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the parameters are
  # loaded from vggish_params.py in the current directory.
  $ python vggish_audio_embeddings.py --wav_path path/to/wav/files/
"""

from __future__ import print_function

import tensorflow.compat.v1 as tf
import os
import pandas as pd
import argparse

import vggish_input
import vggish_params
import vggish_slim

def get_inputs():
  # Define argparse parameters to run from the command line
  parser = argparse.ArgumentParser(description="Process wav files into log mel "+
                                  "spectrograms and feed into VGGish for "+
                                  "embedding generation.")
  parser.add_argument('--wav_path', action='store', required=True,
                      help="Path to directory containing .wav files. "+
                      "Should contain signed 16-bit PCM samples.")
  parser.add_argument('--save_path', action='store', required=True,
                      help="Path to embedding directory. ")
  parser.add_argument('--vggish_checkpoint', action='store', required=False,
                      default= 'vggish_model.ckpt',
                      help="Path to the VGGish checkpoint file. Checkpoints "+
                      "outside the one downloaded using the instructions in the "+
                      "README have not been tested.")
  args = parser.parse_args()
  #Testing for valid user inputs
  if args.wav_path and os.path.exists(args.wav_path):
    wav_path = args.wav_path
  else:
    raise TypeError("Pass in valid path to wav files when calling the function" +
                    "e.g., $ python vggish_audio_embeddings.py "+
                    "--wav_path path/to/wav/files/'")
  if args.save_path and os.path.exists(args.save_path):
      save_path = args.save_path
  else:
    raise TypeError("Pass in valid path to save location for embeddings files " + 
                    "when calling the function" +
                    "e.g., $ python vggish_audio_embeddings.py "+
                    "--save_path path/to/saved/files/'")
  return args


def main(_):
  args = get_inputs()
  wav_path = args.wav_path
  save_path = args.save_path
  vggish_checkpoint = args.vggish_checkpoint
  # Create a list of wav files from the user-input path
  file_list = [wav_path+file for file in os.listdir(wav_path) if file.endswith('.wav')]
  if len(file_list) <= 0:
    raise TypeError("The user must specify a path containing .wav files "+
                    "sampled as signed 16-bit PCM, 16kHz mono.")

  # Define the model in inference mode, load the checkpoint, and
  # locate input and output tensors.
  with tf.Graph().as_default(), tf.Session() as sess:
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, args.vggish_checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  
    for wav_file in file_list:
      # Transform the wav file into log mel spectrograms
      examples_batch = vggish_input.wavfile_to_examples(wav_file)
          
      # Run inference (create embeddings) for the log mel spectrogram
      [embedding_batch] = sess.run([embedding_tensor],
                                  feed_dict={features_tensor: examples_batch})

      # Retrieve the name of the wav file (file name - ".wav")
      wav_filename = wav_file[len(wav_path):-4]

      # Make a dataframe with embeddings and sample information
      embedding_df = pd.DataFrame(embedding_batch)
      embedding_df['wav_filename'] = wav_filename
      embedding_df['example_number'] = range(len(embedding_df))
      embedding_df['recording_start_s'] = (embedding_df['example_number']) * vggish_params.EXAMPLE_WINDOW_SECONDS
      embedding_df['recording_stop_s'] = (embedding_df['example_number'] + 1) * vggish_params.EXAMPLE_WINDOW_SECONDS
      
      # Save the embedding and sample information to a csv file
      embedding_df.to_csv(save_path+wav_filename+'.csv')
      print("{0} embeddings created and saved in 'embedding_data'".format(wav_filename))

if __name__ == '__main__':
  tf.app.run()