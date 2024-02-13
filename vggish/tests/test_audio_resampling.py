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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..")) #puts us in the vggish directory
from scr import audio_resampling


# pylint:disable=unused-argument; is 'magic' argument allowing mock of user input
# pylint:disable=duplicate-code; similar user input testing in vggish test scripts


class TestGetInputs(unittest.TestCase):
    """
    Test suite for audio_resampling arg parse function
    """
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
                save_path = './tests/Embeddings'
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
                wav_path = './tests/',
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
                save_path = './tests/Embeddings'
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
                wav_path = './tests/',
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
                wav_path = './tests/',
                target_sample_rate = -16000,
                save_path = './tests/Embeddings'
            ))
    def test_bad_sample_rate(self, mock_parse_args):
        """
        Test that when the user inputs a sample rate <0 it throws a TypeError
        """
        with self.assertRaises(TypeError):
            audio_resampling.get_inputs()


    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './tests/',
                target_sample_rate = 16000.5,
                save_path = './tests/Embeddings'
            ))
    def test_int_sample_rate(self, mock_parse_args):
        """
        Test that when the sample rate isn't an int it throws a TypeError
        """
        with self.assertRaises(TypeError):
            audio_resampling.get_inputs()


class TestMain(unittest.TestCase):
    """
    Test suite for audio_resampling main function
    """
    @patch('argparse.ArgumentParser.parse_args',
            return_value = argparse.Namespace(
                wav_path = './tests/Embeddings',
                target_sample_rate = 16000,
                save_path = './tests/Embeddings'
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
                wav_path = './tests/',
                target_sample_rate = 16000,
                save_path = './tests/'
            ))
    def test_correct_resampling(self, mock_parse_args):
        """
        Test that the output file from audio_resampling is indeed
        signed 16-bit PCM, sampled as 16kHz mono
        """
        #Running audio_resampling.py
        audio_resampling.main()

        # Checking if the file exists
        sound_file_path = './tests/sample_wav_resampled.wav'
        self.assertTrue(os.path.exists(sound_file_path))

        # Reading the sound file using soundfile
        data, samplerate = sf.read(sound_file_path)

        # Asserting that the data and samplerate are not None
        self.assertIsNotNone(data)
        self.assertIsNotNone(samplerate)

        # Asserting the samplerate is 16000, mono, Wav file, as PCM signed 16-bit
        info = sf.info(sound_file_path)
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.format, 'WAV')
        self.assertEqual(info.subtype, 'PCM_16')

        #Remove the wav file after we're done testing it
        os.remove(sound_file_path)

if __name__ == '__main__':
    unittest.main()
