
# Imports
import csv
from .WordSimilarity import WordSimilarity


# Word similarity dataset
class Wordsim353(WordSimilarity):
    """
    Word Similarity dataset 353
    """

    # Constructor
    def __init__(self, csvreader):
        """
        Constructor
        """
        super(Wordsim353, self).__init__()

        # Words
        self._words = dict()

        # Add words
        for row in csvreader:
            # Info
            word1 = row[0]
            word2 = row[1]

            # Skip header
            if word1 != "Word 1":
                similarity = float(row[2])

                # Create dict for word 1 if necessary
                if word1 not in self._words.keys():
                    self._n_words += 1
                    self._words[word1] = dict()
                # end if

                # Create dict for word 2 if necessary
                if word2 not in self._words.keys():
                    self._n_words += 1
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

    # Get similar words
    def get_similar_words(self, word, limit=10):
        """
        Get similar words
        :param word:
        :param limit:
        :return:
        """
        similar_words = list()

        # For each words
        for word2 in self._words[word].keys():
            similar_words.append((word2, self._words[word][word2]))
        # end for

        # Sort
        similar_words.sort(key=lambda tup: tup[1])

        return similar_words[:limit]
    # end get_similar_words

    # Get list of words
    def words(self):
        """
        Get list of words
        :return:
        """
        return self._words.keys()
    # end words

    ########################################
    # Override
    ########################################

    # Next element
    def next(self):
        """
        Next element
        :return:
        """
        if self._word_pos >= self._n_words:
            self._word_pos = 0
            raise StopIteration()
        else:
            word = self._words.keys()[self._word_pos]
            self._word_pos += 1
            return word, self.get_similar_words(word, limit=1000000)
        # end if
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
        with open(param, 'rb') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            return Wordsim353(csvread)
        # end with
    # end load

# end WordSimilarity
