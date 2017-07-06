
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
    def __init__(self, dim=300, lang='en'):
        """
        Constructor
        """
        # Properties
        self._lang = lang
        self._dim = dim
        self._voc = dict()
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Create a new word vector randomly
    def create_word_vector(self, word):
        """
        Create a new word vector randomly
        :param word: The word
        """
        if word in self._voc.keys():
            raise WordAlreadyExistsException("The word already exists in the vocabulary")
        else:
            self._voc[word] = np.random.random(self._dim)
        # end if
    # end create_word_vector

    # Get a word vector
    def __getattr__(self, item):
        """
        Get a word vector.
        :param item: Item to retrieve, if does not exists, create it.
        :return: The attribute value
        """
        if item not in self._voc:
            self.create_word_vector(item)
        # end if
        return self._voc[item]
    # end __getattr__

    # Set a word vector
    def __setattr__(self, word, vector):
        """
        Set a word vector.
        :param word: Word to set
        :param vector: New word's vector
        """
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
        doc = nlp(text)

        # Resulting numpy array
        doc_array = np.array([])

        # For each word
        for word in doc:
            doc_array = np.vstack((doc_array, self[word]))
        # end for

        return doc_array
    # end __call__

    ###########################################
    # Private
    ###########################################

# end Word2Vec
