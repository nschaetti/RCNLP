#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
#

# Import package
import math
import logging
import numpy as np


# Visualize data
class Visualization:

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        pass
    # end __init__

    @staticmethod
    def similar_words(words, word2vec, distance_measure='euclidian'):
        """
        Similar words
        :param words:
        :param word2vec:
        :param distance_measure:
        :return:
        """
        # For each words
        for word in words:
            logging.info(
                u"Words similar to {} ({}) : {}".format(word, word2vec.get_word_count(word),
                                                        word2vec.get_similar_words(word, distance_measure)))
        # end for
    # end similar_words

    @staticmethod
    def king_man_woman(word2vec, word1, word2, word3, distance_measure='euclidian'):
        """
        king - man + woman
        :param word2vec:
        :param word1:
        :param word2:
        :param word3:
        :param distance_measure:
        :return:
        """
        # Get word vectors
        word1_vec = word2vec.get_word_embeddings_vector(word1)
        word2_vec = word2vec.get_word_embeddings_vector(word2)
        word3_vec = word2vec.get_word_embeddings_vector(word3)

        # Count
        word1_count = word2vec.get_word_count(word1)
        word2_count = word2vec.get_word_count(word2)
        word3_count = word2vec.get_word_count(word3)

        # Compute
        queen = word1_vec - word2_vec + word3_vec

        # Get nearest
        nearest_words = word2vec.nearest_word(queen, distance_measure)

        # Show
        logging.info(
            u"Nearest words to {}({}) - {}({}) + {}({}) : {}".format(word1, word1_count, word2, word2_count, word3,
                                                                     word3_count, nearest_words))
    # end king_man_woman

# end Metrics
