
#Imports
import numpy as np
import spacy

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
    # end __init__

    ###########################################
    # Public
    ###########################################

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
            else:
                self._voc[word] = Word2Vec.sparse(self._dim, self._sparsity)
            # end
        # end if
    # end create_word_vector

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
                doc_array = np.vstack((doc_array, self[word.text]))
            # end if
        # end for

        return doc_array
    # end __call__

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

# end Word2Vec
