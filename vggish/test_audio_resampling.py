"""Code used to test audio embedding extraction from wav files using TensorFlow's
VGGish model.

Input(s):
  # None

Output(s):
  # Validation that testing works

Usage:
  # Validating user conditions are met and throw errors as expected
"""

#For general unittesting
# pylint:disable=import-error; not sure what's going on here
# pylint:disable=no-name-in-module; these names are definitely in the modult - needed for GH
import unittest
import argparse
from unittest.mock import patch
from scr import audio_resampling
#For unittesting main & the outputs of the test audio file
import soundfile as sf

# pylint:disable=unused-argument; is 'magic' argument allowing mock of user input
# pylint:disable=duplicate-code; similar user input testing in vggish test scripts


class TestGetInputs(unittest.TestCase):
    """Test suite for audio_resampling arg parse function"""

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '',
                target_sample_rate = '',
                save_path = ''
            ))
    def test_no_paths(self, mock_parse_args):
        """
        Test that when no paths are input to audio_resampling
        it throws an error
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '',
                target_sample_rate = 16000,
                save_path = '../tests/Embeddings'
            ))
    def test_no_wav_path(self, mock_parse_args):
        """
        Test that when no wav path is input to audio_resampling
        it throws an error
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                target_sample_rate = 16000,
                save_path = ''
            ))
    def test_no_save_path(self, mock_parse_args):
        """
        Test that when no save path is input to audio_resampling
        it throws an error
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './bad_path/',
                target_sample_rate = 16000,
                save_path = '../tests/Embeddings'
            ))
    def test_bad_wav_path(self, mock_parse_args):
        """
        Test that when an invalid wav path is input to audio_resampling
        it throws an error
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                target_sample_rate = 16000,
                save_path = './bad_path/'
            ))
    def test_bad_save_path(self, mock_parse_args):
        """
        Test that when an invalid save path is input to audio_resampling
        it throws an error
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                target_sample_rate = -16000,
                save_path = '../tests/Embeddings'
            ))
    def test_bad_sample_rate(self, mock_parse_args):
        """
        Test that when the user inputs a sample rate <0 it throws a TypeError
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                target_sample_rate = 16000.5,
                save_path = '../tests/Embeddings'
            ))
    def test_int_sample_rate(self, mock_parse_args):
        """
        Test that when the sample rate isn't an int it throws a TypeError
        """
        with self.assertRaises(TypeError):
            audio_resampling()

class TestMain(unittest.TestCase):
    """Test suite for audio_resampling main function"""

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/Embeddings',
                target_sample_rate = 16000,
                save_path = '../tests/Embeddings'
            ))
    def test_no_wav_files(self, mock_parse_args):
        """
        Test that when the user inputs a real path with no wav files to
        audio_resampling it throws an error
        """
        with self.assertRaises(TypeError):
            audio_resampling()

    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = '../tests/',
                target_sample_rate = 16000,
                save_path = '../tests/'
            ))
    def test_correct_resampling(self, mock_parse_args):
        """
        Test that the output file from audio_resampling is indeed
        signed 16-bit PCM, sampled as 16kHz mono
        """
        info = sf.info('tests/sample_wav_resampled.wav')
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.format, 'WAV')
        self.assertEqual(info.subtype, 'PCM_16')


if __name__ == '__main__':
    unittest.main()
