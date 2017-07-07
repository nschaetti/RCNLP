
# Imports
import numpy as np
from .Word2Vec import Word2Vec


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
        input_vectors = self._word2vec(text)
        output_vectors = np.vstack((input_vectors[1:], np.zeros(self._word2vec.get_dimension())))
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

# end WordPredictionDataset
