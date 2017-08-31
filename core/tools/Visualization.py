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
from sklearn.manifold import TSNE
import pylab as plt
from sklearn.decomposition import PCA


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
    def similar_words(words, word2vec, distance_measure='euclidian', limit=10):
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
                                                        word2vec.get_similar_words(word, distance_measure, limit)))
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

    @staticmethod
    def save_figure(word2vec, reduced_matrix, selected_word_indexes, image, fig_size=5000):
        """
        Save figure of words
        :return:
        """
        # Show figure
        plt.figure(figsize=(fig_size * 0.003, fig_size * 0.003), dpi=300)
        max_x = np.amax(reduced_matrix, axis=0)[0]
        max_y = np.amax(reduced_matrix, axis=0)[1]
        min_x = np.amin(reduced_matrix, axis=0)[0]
        min_y = np.amin(reduced_matrix, axis=0)[1]
        plt.xlim((min_x * 1.2, max_x * 1.2))
        plt.ylim((min_y * 1.2, max_y * 1.2))
        for word_text in selected_word_indexes.keys():
            word_count = word2vec.get_word_count(word_text)
            word_index = selected_word_indexes[word_text]
            plt.scatter(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], 0.5)
            plt.text(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1],
                     word_text + u" (" + str(word_count) + u")", fontsize=2.5)
        # end for

        # Save image
        logging.info(u"Saving figure to {}".format(image + ".png"))
        plt.imsave(image + ".png")
    # end save_figure

    @staticmethod
    def top_words_figure(word2vec, word_embeddings, image, fig_size=5000, count_limit=500, reduction='TSNE'):
        """
        Show top words figure
        :param word2vec:
        :param word_embeddings:
        :param image:
        :param fig_size:
        :param reduction:
        :return:
        """
        # Order by word count
        word_counters = list()
        word_counts = word2vec.get_word_counts()
        for word_text in word_counts.keys():
            word_counters.append((word_text, word_counts[word_text]))
        # end for
        word_counters = sorted(word_counters, key=lambda tup: tup[1], reverse=True)

        # Select top-words
        selected_word_embeddings = np.zeros((word_embeddings.shape[0], count_limit))
        selected_word_indexes = dict()
        word_pos = 0
        for (word_text, word_count) in word_counters[: count_limit]:
            word_index = word2vec.get_word_index(word_text)
            selected_word_embeddings[:, word_pos] = word_embeddings[:, word_index]
            selected_word_indexes[word_text] = word_pos
            word_pos += 1
        # end for

        # Word embedding matrix's size
        logging.info(u"Top words, embeddings matrix's size : {}".format(selected_word_embeddings.shape))

        # Reduce with t-SNE
        logging.info(u"Top words, reducing word embedding with {}".format(reduction))
        reduced_matrix = Visualization.reduction(selected_word_embeddings, reduction)

        # Word embedding matrix's size
        logging.info(u"Top words, reduced matrix's size : {}".format(reduced_matrix.shape))

        # Save figure
        Visualization.save_figure(word2vec, reduced_matrix, selected_word_indexes, image, fig_size)
    # end top_words_figure

    @staticmethod
    def words_figure(words, word2vec, word_embeddings, image, fig_size=5000, reduction='TSNE'):
        """
        Show a figure of specific words
        :param word2vec:
        :param word_embeddings:
        :param image:
        :param fig_size:
        :param reduction:
        :return:
        """
        # Selected word embeddings
        selected_word_embeddings = np.zeros((word_embeddings.shape[0], len(words)))

        # For each words
        selected_word_indexes = dict()
        word_pos = 0
        for word in words:
            word_index = word2vec.get_word_index(word)
            selected_word_embeddings[:, word_pos] = word_embeddings[:, word_index]
            selected_word_indexes[word] = word_pos
            word_pos += 1
        # end for

        # Word embedding matrix's size
        logging.info(u"Selected words, embeddings matrix's size : {}".format(selected_word_embeddings.shape))

        # Reduce
        logging.info(u"Selected words, reducing word embedding with {}".format(reduction))
        reduced_matrix = Visualization.reduction(selected_word_embeddings, reduction)

        # Word embedding matrix's size
        logging.info(u"Selected words, reduced matrix's size : {}".format(reduced_matrix.shape))

        # Save figure
        Visualization.save_figure(word2vec, reduced_matrix, selected_word_indexes, image, fig_size)
    # end words_figure

    @staticmethod
    def reduction(word_embeddings, reduction='TSNE'):
        """
        Reduction
        :param word_embeddings:
        :param reduction:
        :return:
        """
        if reduction == 'TNSE':
            return Visualization.reduction_tsne(word_embeddings)
        else:
            return Visualization.reduction_pca(word_embeddings)
        # end if
    # end reduction

    @staticmethod
    def reduction_tsne(word_embeddings):
        """
        Reduction with TSNE
        :return:
        """
        model = TSNE(n_components=2, random_state=0)
        return model.fit_transform(word_embeddings.T)
    # end reduction_tsne

    @staticmethod
    def reduction_pca(word_embeddings):
        """
        Reduction with PCA
        :param word_embeddings:
        :return:
        """
        model = PCA(n_components=2, random_state=0)
        return model.fit_transform(word_embeddings.T)
    # end reduction_pca

# end Metrics
