
# Imports
import numpy as np
import scipy.sparse as sp
import sys


# Word prediction dataset creator
class WordPredictionDataset(object):
    """
    Word prediction dataset creator
    """

    # Constructor
    def __init__(self, word2vec):
        """
        Constructor
        :param word2vec: Word to vec converter
        """
        self._word2vec = word2vec
        self._X = list()
        self._Y = list()
    # end __init__

    # Add an example
    def add(self, text):
        """
        Add an example
        :param text: The text example
        """
        input_vectors = sp.vstack((sp.csr_matrix(np.zeros(self._word2vec.get_dimension())), self._word2vec(text)))
        output_vectors = sp.vstack((self._word2vec(text), sp.csr_matrix(np.zeros(self._word2vec.get_dimension()))))
        self._X.append(input_vectors)
        self._Y.append(output_vectors)
    # end add

    # Get dataset
    def get_dataset(self):
        """
        Get the dataset
        :return:
        """
        return zip(self._X, self._Y)
    # end get_dataset

    # Reset dataset
    def reset(self):
        """
        Reset dataset
        :return:
        """
        self._X = list()
        self._Y = list()
    # end reset

# end WordPredictionDataset
