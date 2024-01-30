"""Code used to test audio embedding extraction from wav files using TensorFlow's
VGGish model.

Input(s):
  # None

Output(s):
  # Validation method that testing works

Usage:
  # Tests that user input a path
"""

import unittest
from vggish_audio_embeddings import get_inputs


class TestGetInputs(unittest.TestCase):
    """Test suite for vggish_audio_embeddings function"""

    def test_no_paths(self):
        """
        Test that when no paths are input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, get_inputs, '', '')

    def test_no_wav_path(self):
        """
        Test that when no wav path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, get_inputs, '',
                          '../tests/Embeddings')
    
    def test_no_save_path(self):
        """
        Test that when no save path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, get_inputs, '../tests/'
                          , '')

    def test_bad_wav_path(self):
        """
        Test that when an invalid wav path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, get_inputs, './bad_path'
                          , '../tests/Embeddings')
    
    def test_bad_save_path(self):
        """
        Test that when an invalid save path is input to vggish_audio_embedding
        it throws an error
        """
        self.assertRaises(TypeError, get_inputs, '../tests/Embeddings',
                          './bad_path')

    def test_no_wav_files(self):
        """
        Test that when the user inputs a real path with no wav files to
        vggish_audio_embedding it throws an error
        """
        self.assertRaises(TypeError, get_inputs, '../tests/Embeddings',
                          '../tests/Embeddings')

if __name__ == '__main__':
    unittest.main()