This README contains information to run the test VGGish implementation

## Steps to Reproduce Analysis
1. Activate the vggish_environment virtual environment
2. Download the "requirements.txt" file from https://github.com/tensorflow/models/tree/master/research/audioset/vggish and move it into the directory you plan to run your code in. For us it's hydropho-natics/poc/vggish.
3. Run "pip install -r requirements.txt" which installs the dependent libraries.
4. *Taken from the TensorFlow VGGish README linked above*
    # Clone TensorFlow models repo into a 'models' directory.
    $ git clone https://github.com/tensorflow/models.git
    $ cd models/research/audioset/vggish
    # Download data files into same directory as code.
    $ curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
    $ curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
5. Execute the smoke test *Taken from the TensorFlow VGGish README linked above*
    # Installation ready, let's test it.
    $ python vggish_smoke_test.py
    # If we see "Looks Good To Me", then we're all set.



## To Dos:
1. Add the items in the requirements.txt file to our environment so the user doesn't have to install them.

## Check out:
Where VGGIsh lives - https://github.com/tensorflow/models/tree/master/research/audioset/vggish
The collab with the how-to - https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF

