#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
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

import argparse
import numpy as np
import os
import io
from core.embeddings.Word2Vec import Word2Vec, OneHotVectorFullException
from core.embeddings.EchoWordPrediction import EchoWordPrediction
from core.embeddings.WordPredictionDataset import WordPredictionDataset
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
import pylab as plt
from sklearn.decomposition import PCA
import logging
import pickle
from core.embeddings.Wordsim353 import Wordsim353
from core.tools.Metrics import Metrics
from core.tools.Visualization import Visualization

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Echo Word Prediction Experience"
ex_instance = "Echo Language Model One Hot Test"

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word prediction with Echo State Network and one-hot vector on Wikipedia")

    # Argument
    parser.add_argument("--file", type=str, help="Word embedding file", required=True)
    parser.add_argument("--image", type=str, help="Output image", default=None, required=False)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    parser.add_argument("--fig-size", type=float, help="Figure size (pixels)", default=1024.0)
    parser.add_argument("--count-limit", type=int, help="Lower limit of word count to display a word", default=50)
    parser.add_argument("--wordsims", type=str, help="Word similarity dataset", required=True)
    parser.add_argument("--n-similar-words", type=int, help="Number of similar words", default=20)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level, format='%(asctime)s :: %(levelname)s :: %(message)s')

    # Print precision
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.nan)

    # Load Wordsim353
    wordsim353 = Wordsim353.load(args.wordsims)

    # Word2Vec
    word2vec = Word2Vec(dim=args.voc_size, mapper='one-hot')

    # Current word embeddings
    word_embeddings = None

    # Word embedding matrix's size
    logging.info(u"Word embedding matrix's size : {}".format(word_embeddings.shape))
    logging.info(u"Word embedding vectors average : {}".format(np.average(word_embeddings)))
    logging.info(u"Word embedding vectors sddev : {}".format(np.std(word_embeddings)))

    # Set word embeddings
    word2vec.set_word_embeddings(word_embeddings=word_embeddings)

    # For each distance measure
    for distance_measure in ['euclidian', 'cosine', 'cosine_abs']:
        print(u"#" * 100)
        print(u"# " + distance_measure)
        print(u"#" * 100)

        # Similarities
        Visualization.similar_words(
            [u"he", u"computer", u"million", u"Toronto", u"France", u"phone", u"ask", u"september", u"blue", u"king",
             u"man", u"woman"],
            word2vec, distance_measure=distance_measure, limit=args.n_similar_words)

        # Word computing
        Visualization.king_man_woman(word2vec, u"king", u"man", u"woman", distance_measure=distance_measure)

        # Test relatedness
        relatedness, relatedness_words = Metrics.relatedness(wordsim353, word2vec, distance_measure=distance_measure)
        print(u"Relatedness : {}, on {} words".format(relatedness, relatedness_words))
    # end for

    # If we want a figure
    if args.image is not None:
        # Order by word count
        word_counters = list()
        word_counts = word2vec.get_word_counts()
        for word_text in word_counts.keys():
            word_counters.append((word_text, word_counts[word_text]))
        # end for
        word_counters = sorted(word_counters, key=lambda tup: tup[1], reverse=True)

        # Select top-words
        selected_word_embeddings = np.zeros((501, args.count_limit))
        selected_word_indexes = dict()
        word_pos = 0
        for (word_text, word_count) in word_counters[: args.count_limit]:
            word_index = word2vec.get_word_index(word_text)
            selected_word_embeddings[:, word_pos] = word_embeddings[:, word_index]
            selected_word_indexes[word_text] = word_pos
            word_pos += 1
        # end for

        # Word embedding matrix's size
        logging.info(u"Selected word embeddings matrix's size : {}".format(selected_word_embeddings.shape))

        # Reduce with t-SNE
        logging.info(u"Reducing word embedding with TSNE")
        model = TSNE(n_components=2, random_state=0)
        reduced_matrix = model.fit_transform(selected_word_embeddings.T)

        # Word embedding matrix's size
        logging.info(u"Reduced matrix's size : {}".format(reduced_matrix.shape))

        # Show t-SNE
        plt.figure(figsize=(args.fig_size*0.003, args.fig_size*0.003), dpi=300)
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
            plt.text(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], word_text + u" (" + str(word_count) + u")", fontsize=2.5)
        # end for

        # Save image
        logging.info(u"Saving figure to {}".format(args.image + str(loop) + ".png"))
        plt.savefig(args.image + str(loop) + ".png")
    # end if

# end if
