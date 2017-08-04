
#Imports
import numpy as np
import spacy
from numpy import linalg as LA
import scipy.sparse as sp

###########################################################
# Exceptions
###########################################################


# The word already exists in the vocabulary
class WordAlreadyExistsException(Exception):
    """
    The word already exists in the vocabulary
    """
    pass
# end WordAlreadyExistsException


# One-hot vector representations are full
class OneHotVectorFullException(Exception):
    """
    One-hot vector representations are full
    """
    pass
# end OneHotVectorFullException

###########################################################
# Class
###########################################################


# Word to Dense vector converters
class Word2Vec(object):
    """
    Word to Dense Vector converters
    """

    # Constructor
    def __init__(self, dim=300, lang='en', mapper='dense', sparsity=0.02):
        """
        Constructor
        """
        # Properties
        self._lang = lang
        self._dim = dim
        self._voc = dict()
        self._mapper = mapper
        self._sparsity = sparsity
        self._word_pos = 0
        self._word_index = dict()
        self._index_word = dict()
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Get mapper
    def get_mapper(self):
        """
        Get mapper
        :return:
        """
        return self._mapper
    # end get_mapper

    # Get dimension
    def get_dimension(self):
        """
        Get dimension
        :return:
        """
        return self._dim
    # end get_dimension

    # Get lang
    def get_lang(self):
        """
        Get lang.
        :return:
        """
        return self._lang
    # end get_lang

    # Words
    def words(self):
        """
        Words
        :return:
        """
        return self._voc.keys()
    # end words

    # Create a new word vector randomly
    def create_word_vector(self, word):
        """
        Create a new word vector randomly
        :param word: The word
        """
        if word in self._voc.keys():
            raise WordAlreadyExistsException("The word already exists in the vocabulary")
        else:
            if self._mapper == "dense":
                self._voc[word] = Word2Vec.dense(self._dim)
            elif self._mapper == "sparse":
                self._voc[word] = Word2Vec.sparse(self._dim, self._sparsity)
            elif self._mapper == "one-hot":
                if self._word_pos < self._dim:
                    self._voc[word] = self._one_hot()
                    self._word_index[word] = self._word_pos - 1
                    self._index_word[self._word_pos-1] = word
                else:
                    raise OneHotVectorFullException("One-hot vector representations are full")
                # end if
            # end
        # end if
    # end create_word_vector

    # Normalize each vector
    def normalize(self):
        """
        Normalize each vector
        """
        for word in self._voc.keys():
            # print(self._voc[word])
            # print(LA.norm(self._voc[word]))
            self._voc[word] /= LA.norm(self._voc[word])
        # end for
    # end normalize

    # Get matrix
    def get_matrix(self):
        """
        Get matrix
        :return:
        """
        words_matrix = np.zeros((len(self._voc.keys()), self._dim))
        for index, word in enumerate(self.words()):
            words_matrix[index, :] = self._voc[word]
        # end for
        return words_matrix
    # end get_matrix

    # Get word by index
    def get_word_by_index(self, index):
        """
        Get word by index
        :param index: Index
        :return:
        """
        if index < len(self._index_word):
            return self._index_word[index]
        else:
            return None
        # end if
    # end get_word_by_index

    # Get word count
    def get_n_words(self):
        """
        Get word count
        :return: word count
        """
        return self._word_pos
    # end get_n_words

    ###########################################
    # Override
    ###########################################

    # Get a word vector
    def __getitem__(self, item):
        """
        Get a word vector.
        :param item: Item to retrieve, if does not exists, create it.
        :return: The attribute value
        """
        item = item.lower()
        if item not in self._voc.keys():
            self.create_word_vector(item)
        # end if
        return self._voc[item]
    # end __getattr__

    # Set a word vector
    def __setitem__(self, word, vector):
        """
        Set a word vector.
        :param word: Word to set
        :param vector: New word's vector
        """
        word = word.lower()
        self._voc[word] = vector
    # end if

    # Transform text to matrix
    def __call__(self, text):
        """
        Transform test to matrix
        :param text: Text to transform.
        :return: Matrix representation of the text.
        """
        # Load language model
        nlp = spacy.load(self._lang)

        # Process text
        doc = nlp(text.lower())

        # Resulting numpy array
        doc_array = np.array([])

        # For each word
        for word in doc:
            if doc_array.size == 0:
                doc_array = self[word.text]
            else:
                if self._mapper == "one-hot":
                    doc_array = sp.vstack(blocks=[doc_array, self[word.text]])
                else:
                    doc_array = np.vstack((doc_array, self[word.text]))
                # end if
            # end if
        # end for

        return doc_array
    # end __call__

    # Left multiplication
    def __mul__(self, other):
        """
        Left multiplication
        :param other:
        :return:
        """
        for word in self._voc.keys():
            self._voc[word] *= other
        # end for
        return self
    # end __mul__

    # Right multiplication
    def __rmul__(self, other):
        """
        Right multiplication
        :param other:
        :return:
        """
        for word in self._voc.keys():
            self._voc[word] *= other
        # end for
        return self
    # end __rmul__

    # Augmented assignment (mult)
    def __imul__(self, other):
        """
        Augmented assignment (mult)
        :param other:
        :return:
        """
        for word in self._voc.keys():
            self._voc[word] *= other
        # end for
        return self
    # end __imul__

    ###########################################
    # Private
    ###########################################

    # Map word to a dense vector
    @staticmethod
    def dense(dim):
        """
        Map word to a dense mapper
        :param dim: Vector dimension
        :return: A new dense vector
        """
        return np.random.random(dim) * 2.0 - 1.0
    # end _dense_mapper

    # Map word to a sparse vector
    @staticmethod
    def sparse(dim, sparsity):
        """
        Map word to a sparse mapper
        :param dim: Vector dimension
        :param sparsity: Vector sparsity
        :return: A new dense vector
        """
        vec = np.zeros(dim)
        vec[np.random.random(dim) > (1.0 - sparsity)] = 1.0
        return vec
    # end _dense_mapper

    # Map word to a one-hot vector
    def _one_hot(self):
        """
        Map word to a one-hot vector
        :return: A new one-hot vector
        """
        vec = np.zeros(self._dim, dtype='float64')
        vec[self._word_pos] = 1.0
        vec = sp.csr_matrix(vec)
        self._word_pos += 1
        return vec
    # end one_hot

# end Word2Vec
