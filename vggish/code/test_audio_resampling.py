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
import audio_resampling


class TestGetInputs(unittest.TestCase):
    """Test suite for vggish_audio_embeddings function"""

    def test_no_paths(self):
        """
        Test that when no paths are input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, audio_resampling, '', '')

    def test_no_wav_path(self):
        """
        Test that when no wav path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, audio_resampling, '',
                          '../tests/Embeddings/')
    
    def test_no_save_path(self):
        """
        Test that when no save path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, audio_resampling, '../tests/'
                          , '')

    def test_bad_wav_path(self):
        """
        Test that when an invalid wav path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, audio_resampling, './bad_path/'
                          , '../tests/Embeddings/')
    
    def test_bad_save_path(self):
        """
        Test that when an invalid save path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, audio_resampling, '../tests/Embeddings/',
                          './bad_path/')

    def test_no_wav_files(self):
        """
        Test that when the user inputs a real path with no wav files to
        vggish_audio_embedding it throws an error
        """
        self.assertRaises(TypeError, audio_resampling, '../tests/Embeddings/',
                          '../tests/Embeddings/')

    def test_bad_sample_rate(self):
        """
        Test that when the user inputs a sample rate <0 it throws a TypeError
        """
        self.assertRaises(TypeError, audio_resampling, '../tests/',
                          '../tests/', target_sample_rate = -16000)

if __name__ == '__main__':
    unittest.main()