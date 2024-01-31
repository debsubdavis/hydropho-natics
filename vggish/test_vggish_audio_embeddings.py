"""Code used to test audio embedding extraction from wav files using TensorFlow's
VGGish model.

Input(s):
  # None

Output(s):
  # Validation that testing works

Usage:
  # Validating user conditions are met and throw errors as expected
"""

#General unittesting
import unittest
import argparse
from unittest.mock import patch
from code.vggish_audio_embeddings import main
from code.vggish_audio_embeddings import get_inputs
import pandas as pd
import soundfile as sf


# pylint:disable=unused-argument; is 'magic' argument allowing mock of user input


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
            get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = '../tests/Embeddings'
            ))
    def test_no_wav_path(self, mock_parse_args):
        """
        Test that when no wav path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = ''
            ))
    def test_no_save_path(self, mock_parse_args):
        """
        Test that when no save path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            get_inputs()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './bad_path/',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = '../tests/Embeddings'
            ))
    def test_bad_wav_path(self, mock_parse_args):
        """
        Test that when an invalid wav path is input to vggish_audio_embedding
        it throws an error
        """
        with self.assertRaises(TypeError):
            get_inputs()

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
            get_inputs()

class TestMain(unittest.TestCase):
    """Test suite for vggish_audio_embeddings main function"""

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './bad_path',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = '../tests/Embeddings'
            ))
    def test_no_wav_files(self, mock_parse_args):
        """
        Test that when the user inputs a real path with no wav files to
        vggish_audio_embedding it throws an error
        """
        with self.assertRaises(TypeError):
            main()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                vggish_checkpoint = 'vggish_model.ckpt',
                save_path = '../tests/Embeddings'
            ))
    def test_csv_len(self, mock_parse_args):
        """
        Test that the output csv is the right length based on the audio file
        """
        output_csv = pd.read_csv('tests/Embeddings/sample_wav_resampled.csv')
        info = sf.info('tests/sample_wav_resampled.wav')
        wav_file_length = (info.duration)/0.96
        self.assertEqual(len(output_csv), int(wav_file_length))


if __name__ == '__main__':
    unittest.main()
