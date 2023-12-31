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

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

#EKRC adds
import pandas as pd
import glob

flags = tf.app.flags

#Removed the WAV file option in favor of a for loop which will automatically run all
# the WAV files in our intermediate_data directory
flags.DEFINE_string(
    'wav_file', '../../../../intermediate_data/181204-203002-437599-806141979.wav',
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def main(_):
  # EKRC In this POC, we run all wav POC files through the model
  path = r'../../../../intermediate_data/*.wav'
  file_list = glob.glob(path)
  format_file_list = [file.replace('\\', '/') for file in file_list]
  #print(format_file_list)

  for file in format_file_list:
    wav_file = file
    #EKRC we know we have wav files, so we comment out the warning here
    '''    if FLAGS.wav_file:
        wav_file = FLAGS.wav_file
    else:
        print("WAV file must be provided")'''
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    #print(examples_batch) #EKRC

    #EKRC Getting the wav filename
    wav_filename = wav_file[30:-4]

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        #EKRC Run inference but not postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                    feed_dict={features_tensor: examples_batch})
        #print(embedding_batch)

        #EKRC Save the embeddings to a pandas df then csv file
        embedding_df = pd.DataFrame(embedding_batch)
        embedding_df['recording_file'] = wav_filename
        embedding_df['example_number'] = range(len(embedding_df))
        embedding_df['start_time_s'] = (embedding_df['example_number']) * 0.96 #MAKE REUSABLE IN FUTURE
        embedding_df['stop_time_s'] = (embedding_df['example_number'] + 1) * 0.96 #MAKE REUSABLE IN FUTURE 
        #print(embedding_df[['example_number','start_time_s','stop_time_s']])
        
        #EKRC reordering the col of the df to have identification info first, 128D last
        ref_col = embedding_df.pop('recording_file')
        embedding_df.insert(0, 'recording_file',ref_col)
        example_number = embedding_df.pop('example_number')
        embedding_df.insert(1, 'example_number',example_number)
        start_col = embedding_df.pop('start_time_s')
        embedding_df.insert(2, 'start_time_s',start_col)
        stop_col = embedding_df.pop('stop_time_s')
        embedding_df.insert(3, 'stop_time_s',stop_col)
        
        #EKRC saving as a csv
        embedding_df.to_csv('../../../../embedding_data/'+wav_filename+'.csv')
        print("File embeddings created and saved in 'embedding_data'")

if __name__ == '__main__':
  tf.app.run()