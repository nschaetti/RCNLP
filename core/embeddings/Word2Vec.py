#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Imports
import numpy as np
import spacy
import pickle
import re
from numpy import linalg as LA
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from .WordSimilarity import WordSimilarity

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
class Word2Vec(WordSimilarity):
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
        self._word_counter = dict()
        self._total_counter = 0
        self._word_embeddings = np.array([])
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

    # Get word count
    def get_word_count(self, word):
        """
        Get word count
        :param word:
        :return:
        """
        try:
            return self._word_counter[word.lower()]
        except KeyError:
            return 0
        # end try
    # end get_word_count

    # Get word counts
    def get_word_counts(self):
        """
        Get word counts
        :return:
        """
        return self._word_counter
    # end get_word_counts

    # Reset word count
    def reset_word_count(self):
        """
        Reset word count
        :return:
        """
        self._word_counter = dict()
        self._total_counter = 0
    # end if

    # Get total word count
    def get_total_count(self):
        """
        Get total word count
        :return:
        """
        return self._total_counter
    # end get_total_count

    # Get word embeddings
    def get_word_embeddings(self):
        """
        Get word embeddings
        :return:
        """
        return self._word_embeddings
    # end get_word_embeddings

    # Get nearest word
    def nearest_word(self, word_embedding_vector, measure='euclidian', limit=10):
        """
        Get the nearest word from a vector
        :param word_embedding_vector:
        :param measure:
        :return:
        """
        reverse = {'euclidian': False, 'cosine': True, 'cosine_abs': True}

        similarities = list()
        for word in self._voc.keys():
            word_index = self._word_index[word]
            word_vector = self._word_embeddings[:, word_index]
            word_vector = word_vector.reshape(1, -1)
            word_embedding_vector = word_embedding_vector.reshape(1, -1)
            if measure == 'euclidian':
                similarities.append((word, euclidean(word_embedding_vector, word_vector)))
            elif measure == 'cosine_abs':
                similarities.append((word, np.abs(cosine_similarity(word_embedding_vector, word_vector))))
            else:
                similarities.append((word, cosine_similarity(word_embedding_vector, word_vector)))
            # end if
        # end for

        # Sort
        similarities.sort(key=lambda tup: tup[1], reverse=reverse[measure])

        return similarities[:limit]
    # end nearest_word

    # Get word similarities
    def similarity(self, word1, word2, measure='euclidian'):
        """
        Get word similarities
        :param word1:
        :param word2:
        :param measure:
        :return:
        """
        word1 = word1.lower()
        word2 = word2.lower()
        if word1 == word2:
            return 1.0
        elif word1 not in self._voc.keys() or word2 not in self._voc.keys():
            return 0
        else:
            word1_index = self._word_index[word1]
            word2_index = self._word_index[word2]
            word1_vector = self._word_embeddings[:, word1_index]
            word2_vector = self._word_embeddings[:, word2_index]
            word1_vector = word1_vector.reshape(1, -1)
            word2_vector = word2_vector.reshape(1, -1)
            if measure == 'euclidian':
                return euclidean(word1_vector, word2_vector)
            elif measure == 'cosine_abs':
                return np.abs(cosine_similarity(word1_vector, word2_vector))
            else:
                return cosine_similarity(word1_vector, word2_vector)
            # end if
        # end if
    # end similarity

    # Get similar words
    def get_similar_words(self, word1, measure='euclidian', limit=10):
        """
        Get similar words
        :param word1:
        :param measure:
        :param limit:
        :return:
        """
        reverse = {'euclidian': False, 'cosine': True, 'cosine_abs': True}

        word1 = word1.lower()
        similarities = list()
        for word2 in self._voc.keys():
            if word1 != word2:
                similarities.append((word2, self.similarity(word1, word2, measure)))
            # end if
        # end for

        # Sort
        similarities.sort(key=lambda tup: tup[1], reverse=reverse[measure])

        return similarities[:limit]
    # end get_similar_word

    # Set word embeddings
    def set_word_embeddings(self, word_embeddings):
        """
        Set word embeddings
        :param word_embeddings:
        :return:
        """
        self._word_embeddings = word_embeddings
    # end set_word_embeddings

    # Set word indexes
    def set_word_indexes(self, word_indexes):
        """
        Set word indexes
        :param word_indexes:
        :return:
        """
        self._word_index = word_indexes

    # Get word index
    def get_word_index(self, word_text):
        """
        Get word index
        :param word_text:
        :return:
        """
        return self._word_index[word_text]
    # end get_word_index

    # Get word indexes
    def get_word_indexes(self):
        """
        Get word indexes
        :return:
        """
        return self._word_index
    # end get_word_indexes

    # Get word embeddings vector
    def get_word_embeddings_vector(self, word_text):
        """
        Get word embeddings vector
        :param word_text:
        :return:
        """
        return self._word_embeddings[:, self.get_word_index(word_text)]
    # end get_word_embeddings_vector

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
            # Clean
            word_text = word.text
            word_text = word_text.replace(u"\n", u"")
            word_text = word_text.replace(u"\t", u"")
            word_text = word_text.replace(u"\r", u"")
            word_text = word_text.replace(u" ", u"")
            word_text = word_text.replace(u"–", u"-")

            # Replacement
            word_text = Word2Vec.replace_token(word_text, r"^[0-9]{4}\-[0-9]{4}$", u"<interval>")
            word_text = Word2Vec.replace_token(word_text, r"^[0-9]{4}\-[0-9]{2}$", u"<interval>")
            word_text = Word2Vec.replace_token(word_text, r"^\d+th$", u"<th>")
            word_text = Word2Vec.replace_token(word_text, r"^\d+nd$", u"<th>")
            word_text = Word2Vec.replace_token(word_text, r"^[+-]?\d+(?:\.\d+)?\%$", u"<percent>")
            word_text = Word2Vec.replace_token(word_text, r"^[+-]?\d+(?:\.\d+)+$", u"<float>")
            word_text = Word2Vec.replace_token(word_text, r'^\d+(?:,\d+)+$', u"<number>")
            word_text = Word2Vec.replace_token(word_text, r"^[0-9]{4}$", u"<4digits>")
            word_text = Word2Vec.replace_token(word_text, r"^[0-9]{3}$", u"<3digits>")
            word_text = Word2Vec.replace_token(word_text, r"^[0-9]{2}$", u"<2digits>")
            word_text = Word2Vec.replace_token(word_text, r"^[0-9]{1}$", u"<1digit>")
            word_text = Word2Vec.replace_token(word_text, r"^[+-]?\d+$", u"<integer>")

            # Add
            if len(word_text) > 0:
                self._total_counter += 1
                try:
                    self._word_counter[word_text] += 1
                except KeyError:
                    self._word_counter[word_text] = 1
                # end try
                if doc_array.size == 0:
                    doc_array = self[word_text]
                else:
                    if self._mapper == "one-hot":
                        doc_array = sp.vstack(blocks=[doc_array, self[word_text]])
                    else:
                        doc_array = np.vstack((doc_array, self[word_text]))
                    # end if
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

    # Save Word2Vec
    def save(self, file_name):
        """
        Save Word2Vec
        :param file_name:
        :return:
        """
        pickle.dump(self, open(file_name, 'wb'))
    # end save

    ###########################################
    # Static
    ###########################################

    # Replace token if match a regex
    @staticmethod
    def replace_token(token, regex, repl):
        """
        Replace token if match a regex
        :param token:
        :param regex:
        :param repl:
        :return:
        """
        if re.match(regex, token):
            return repl
        # end if
        return token
    # end replace_token

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
