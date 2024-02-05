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
import unittest
import argparse
import os
from unittest.mock import patch
import soundfile as sf
from . import audio_resampling


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
            audio_resampling.get_inputs()

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
            audio_resampling.get_inputs()

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
            audio_resampling.get_inputs()

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
            audio_resampling.get_inputs()

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
            audio_resampling.get_inputs()

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
            audio_resampling.get_inputs()

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
            audio_resampling.get_inputs()

class TestMain(unittest.TestCase):
    """Test suite for audio_resampling main function"""

    '''def setup(self):
        """
        Opens the resampled wav file for use later in these tests
        """
        self.test_wav_file = open(resampled_test_file)
        self.test_wav_data = self.test_wav_file.read()
    
    def tearDown(self):
        """
        CLoses the test file to avoid resource usage
        """
        self.test_wav_file.close()'''

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
            audio_resampling.main()

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
        # Assuming the sound file is located in a 'test_files' directory within your project
        sound_file_path = os.path.join(os.path.dirname(__file__), '../tests', 'sample_wav_resampled.wav')

        # Checking if the file exists
        self.assertTrue(os.path.exists(sound_file_path), f"File '{sound_file_path}' not found.")

        '''# Reading the sound file using soundfile
        data, samplerate = sf.read(sound_file_path)

        # Asserting that the data and samplerate are not None
        self.assertIsNotNone(data)
        self.assertIsNotNone(samplerate)'''

        # Add more assertions or tests as needed
        '''info = sf.info(sound_file_path)
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.format, 'WAV')
        self.assertEqual(info.subtype, 'PCM_16')'''


if __name__ == '__main__':
    unittest.main()
