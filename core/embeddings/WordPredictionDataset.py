
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
    def __init__(self, word2vec, before=0, after=1):
        """
        Constructor
        :param word2vec: Word to vec converter
        :param task_type: Task type (predict, remember, predict_and_remember)
        """
        self._word2vec = word2vec
        self._before = before
        self._after = after
        self._X = list()
        self._Y = list()
    # end __init__

    # Add an example
    def add(self, text):
        """
        Add an example
        :param text: The text example
        """
        # Current word vector
        word_vectors = self._word2vec(text)

        # Before zero vector
        before_zero_vector = sp.csr_matrix(np.zeros((self._before, self._word2vec.get_dimension())))
        after_zero_vector = sp.csr_matrix(np.zeros((self._after, self._word2vec.get_dimension())))

        # Add to dataset
        if self._after != 0 and self._before == 0:
            input_vectors = sp.vstack((after_zero_vector, word_vectors))
            output_vectors = sp.vstack((word_vectors, after_zero_vector))
        elif self._before != 0 and self._after == 0:
            input_vectors = sp.vstack((word_vectors, before_zero_vector))
            output_vectors = sp.vstack((before_zero_vector, word_vectors))
        else:
            # Inputs
            input_vectors = sp.vstack((after_zero_vector, word_vectors, before_zero_vector))
            # Remember
            remember_vectors = sp.vstack((after_zero_vector, before_zero_vector, word_vectors))
            # Predict
            predict_vectors = sp.vstack((word_vectors, after_zero_vector, before_zero_vector))
            # Output
            output_vectors = sp.hstack((remember_vectors, predict_vectors))
        # end if

        # Add to dataset
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
