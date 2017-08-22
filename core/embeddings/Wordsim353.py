
# Imports
import csv


# Word similarity dataset
class Wordsim353(object):
    """
    Word Similarity dataset 353
    """

    # Constructor
    def __init__(self, csvreader):
        """
        Constructor
        """
        # Words
        self._words = dict()

        # Add words
        for row in csvreader:
            # Info
            word1 = row[0]
            word2 = row[1]
            similarity = float(row[2])

            # Skip header
            if word1 != "Word 1":
                # Create dict for word 1 if necessary
                if word1 not in self._words.keys():
                    self._words[word1] = dict()
                # end if

                # Create dict for word 2 if necessary
                if word2 not in self._words.keys():
                    self._words[word2] = dict()
                # end if

                # Add similarity
                self._words[word1][word2] = similarity
                self._words[word2][word1] = similarity
            # end if
        # end for
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
        if word1 in self._words.keys() and word2 in self._words[word1].keys():
            return self._words[word1][word2]
        # end if

        return None
    # end similarity

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
        with open(param, 'rb') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            return Wordsim353(csvread)
        # end with
    # end load

# end WordSimilarity
