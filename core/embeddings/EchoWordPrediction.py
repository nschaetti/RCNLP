
#Imports
import numpy as np
import spacy
import Oger
import mdp
import spacy
from .WordPredictionDataset import WordPredictionDataset

###########################################################
# Exceptions
###########################################################


# Network net trained
class ReservoirNotTrainedException(Exception):
    """
    Network net trained
    """
    pass
# end ReservoirNotTrainedException

###########################################################
# Class
###########################################################


# Echo State Network to predict the next word
class EchoWordPrediction(object):
    """
    Echo State Network to predict the next word
    """

    # Constructor
    def __init__(self, word2vec, size, leaky_rate, spectral_radius, input_scaling=0.25, input_sparsity=0.1,
                 w_sparsity=0.1, w_in=None, w=None, use_sparse_matrix=False, task_type='predict'):
        """
        Constructor
        :param word2vec:
        :param size:
        :param leaky_rate:
        :param spectral_radius:
        :param input_scaling:
        :param input_sparsity:
        :param w_sparsity:
        :param w:
        :param task_type:
        """
        # Properties
        self._word2vec = word2vec
        self._size = size
        self._leaky_rate = leaky_rate
        self._spectral_radius = spectral_radius
        self._trained = False
        self._task_type = task_type

        # Wordprediction dataset generator
        self._dataset = WordPredictionDataset(word2vec=word2vec, task_type=task_type)

        # Dimension
        if self._task_type == 'predict' or self._task_type == 'remember':
            input_dim = word2vec.get_dimension()
        else:
            input_dim = word2vec.get_dimension() * 2
        # end if

        # Create the reservoir
        self._reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=self._size,
                                                        input_scaling=input_scaling,
                                                        leak_rate=leaky_rate, spectral_radius=spectral_radius,
                                                        sparsity=input_sparsity, w_sparsity=w_sparsity,
                                                        w_in=w_in, use_sparse_matrix=use_sparse_matrix)

        # Reset state at each call
        self._reservoir.reset_states = True

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=1)
    # end __init__

    # Change W_in
    def set_w_in(self, w_in):
        """
        Change W_in
        :param w_in:
        :return:
        """
        self._reservoir.w_in = w_in
    # end set_w_in

    # Add text example
    def add(self, text):
        """
        Add text example
        :param text:
        :return:
        """
        self._dataset.add(text)
    # end add

    # Train the reservoir
    def train(self):
        """
        Train the reservoir
        """
        # Create data
        data = [None, self._dataset.get_dataset()]

        # Train the model
        self._flow.train(data)

        # Trained
        self._trained = True
    # end train

    # Predict the next word
    def predict(self, text):
        """
        Predict the next work
        :param text:
        :return:
        """
        # Current word vector
        current_vectors = self._word2vec(text)

        # Predict text
        predicted_words = self._flow(current_vectors)

        # Load language model
        nlp = spacy.load(self._word2vec.get_lang())

        # Process text
        doc = nlp(text.lower())

        # New word vectors
        new_vectors = dict()

        # For each word
        for index, word in enumerate(doc):
            # Add new vector
            if word.text not in new_vectors.keys():
                new_vectors[word.text] = predicted_words[index]
            else:
                new_vectors[word.text] = np.vstack((new_vectors[word.text], predicted_words[index]))
            # end if
        # end for

        return new_vectors
    # end predict

    # Get word embeddings
    def get_word_embeddings(self):
        """
        Get word embeddings
        :return:
        """
        if self._trained:
            return self._readout.beta
        else:
            raise ReservoirNotTrainedException(u"Reservoir not trained!")
        # end if
    # end get_word_embeddings

    # Reset learning but keep reservoir
    def reset(self):
        """
        Reset learning but keep reservoir
        :return:
        """
        del self._readout, self._flow

        # Reset dataset
        self._dataset.reset()

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=1)
    # end reset

# end EchoWordPrediction
