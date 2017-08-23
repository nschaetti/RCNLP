

# Word similarity abstract class
class WordSimilarity(object):
    """
    Word similarity abstract class
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        self._word_pos = 0
        self._n_words = 0
    # end __init__

    ########################################
    # Public
    ########################################

    # Get word similarity
    def similarity(self, word1, word2):
        """
        Get word similarity
        :param word1:
        :param word2:
        :return:
        """
        pass
    # end similarity

    # Get similar words
    def get_similar_words(self, word, limit=10):
        """
        Get similar words
        :param word:
        :param limit:
        :return:
        """
        pass
    # end get_similar_words

    # Get list of words
    def words(self):
        """
        Get list of words
        :return:
        """
        pass
    # end words

    ########################################
    # Override
    ########################################

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next element
    def next(self):
        """
        Next element
        :return:
        """
        pass
    # end next

    ########################################
    # Static
    ########################################

    # Load word similarity
    @staticmethod
    def load(param):
        """
        Load word similarity
        :param param:
        :return:
        """
        pass
    # end load

# end WordSimilarity
