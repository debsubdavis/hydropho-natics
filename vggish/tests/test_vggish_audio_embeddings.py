"""Code used to test audio embedding extraction from wav files using TensorFlow's
VGGish model.

Input(s):
  # None

Output(s):
  # Validation that testing works

Usage:
  # Validating user conditions are met and throw errors as expected
"""

import unittest
import argparse
from unittest.mock import patch
import os
import pandas as pd
import soundfile as sf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scr import vggish_audio_embeddings


# pylint:disable=unused-argument; is 'magic' argument allowing mock of user input
# pylint:disable=duplicate-code; similar user input testing required in vggish test scripts


class TestGetInfo(unittest.TestCase):
    """Test suite for vggish_audio_embeddings arg parsing"""

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = ''
            ))
    def test_no_paths(self, mock_parse_args):
        """
        Test that when no paths are input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            vggish_audio_embeddings.get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = './vggish/tests/'
            ))
    def test_no_wav_path(self, mock_parse_args):
        """
        Test that when no wav path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            vggish_audio_embeddings.get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './vggish/tests/',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = ''
            ))
    def test_no_save_path(self, mock_parse_args):
        """
        Test that when no save path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            vggish_audio_embeddings.get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './bad_path/',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = './vggish/tests/'
            ))
    def test_bad_wav_path(self, mock_parse_args):
        """
        Test that when an invalid wav path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            vggish_audio_embeddings.get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = './bad_path/'
            ))
    def test_bad_save_path(self, mock_parse_args):
        """
        Test that when an invalid save path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            vggish_audio_embeddings.get_inputs()

class TestMain(unittest.TestCase):
    """Test suite for vggish_audio_embeddings main function"""

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './bad_path',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = './vggish/tests/'
            ))
    def test_no_wav_files(self, mock_parse_args):
        """
        Test that when the user inputs a real path with no wav files to
        audio_resampling it throws an error
        """
        with self.assertRaises(TypeError):
            vggish_audio_embeddings.main(mock_parse_args)

    #The following code is used for local testing only
    '''patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './vggish/tests/',
                vggish_checkpoint = './vggish/scr/vggish_model.ckpt',
                save_path = './vggish/tests/'
            ))
    def test_csv_len(self, mock_parse_args):
        """
        Test that the output csv is the right length based on the audio file
        """
        #Running the model
        vggish_audio_embeddings.main(None)

        #Checking if the file exists
        embedding_file_path = './vggish/tests/sample_wav.csv'
        self.assertTrue(os.path.exists(embedding_file_path))

        #Checking that the file is of the appropriate length
        output_csv = pd.read_csv('./vggish/tests/sample_wav.csv')
        info = sf.info('./vggish/tests/sample_wav.wav')
        wav_file_length = (info.duration)/0.96
        self.assertEqual(len(output_csv), int(wav_file_length))

        #Remove the csv file after we're done testing it
        os.remove(embedding_file_path)'''


if __name__ == '__main__':
    unittest.main()