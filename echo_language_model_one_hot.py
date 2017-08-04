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
from core.embeddings.Word2Vec import Word2Vec
from core.embeddings.EchoWordPrediction import EchoWordPrediction
from core.embeddings.WordPredictionDataset import WordPredictionDataset
from numpy import linalg as LA
from sklearn.manifold import TSNE
import pylab as plt
from sklearn.decomposition import PCA
import logging

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Echo Word Prediction Experience"
ex_instance = "Echo Language Model One Hot"

# Reservoir Properties
rc_leak_rate = 0.5  # Leak rate
rc_input_scaling = 1.0  # Input scaling
rc_size = 300  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.01

####################################################
# Functions
####################################################

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word prediction with Echo State Network and one-hot vector")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--output", type=str, help="Output image", required=True)
    parser.add_argument("--size", type=int, help="How many file to take in the dataset", default=-1)
    parser.add_argument("--sparse", action='store_true', help="Sparse matrix?", default=False)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    parser.add_argument("--voc-size", type=int, help="Vocabulary size", default=5000, required=True)
    parser.add_argument("--loop", type=int, help="Number of loops", default=1)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="RCNLP")

    # Print precision
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.nan)

    # Word2Vec
    word2vec = Word2Vec(dim=args.voc_size, mapper='one-hot')

    # ESN for word prediction
    esn_word_prediction = EchoWordPrediction(word2vec=word2vec, size=rc_size, leaky_rate=rc_leak_rate,
                                             spectral_radius=rc_spectral_radius, input_scaling=rc_input_scaling,
                                             input_sparsity=rc_input_sparsity, w_sparsity=rc_w_sparsity,
                                             use_sparse_matrix=args.sparse)

    # For each loop
    for loop in range(args.loop):
        # Add text examples
        for index, file in enumerate(os.listdir(args.dataset)):
            if args.size != -1 and index >= args.size:
                break
            # end if
            file_path = os.path.join(args.dataset, file)
            logger.info(u"Adding text file {}/{} : {}".format(index+1, args.size, file_path))
            esn_word_prediction.add(io.open(file_path, 'r').read())
            logger.info(u"{} total token in word2vec".format(word2vec.get_n_words()))
        # end for

        # Train
        logger.info(u"Training...")
        esn_word_prediction.train()

        # Get word embeddings
        word_embeddings = esn_word_prediction.get_word_embeddings()

        # Word embedding matrix's size
        logger.info(u"Word embedding matrix's size : {}".format(word_embeddings.shape))

        # Reduce with t-SNE
        model = TSNE(n_components=2, random_state=0)
        reduced_matrix = model.fit_transform(word_embeddings.T)

        # Word embedding matrix's size
        logger.info(u"Reduced matrix's size : {}".format(reduced_matrix.shape))

        # Show t-SNE
        plt.figure(figsize=(15, 15), dpi=300)
        max_x = np.amax(reduced_matrix, axis=0)[0]
        max_y = np.amax(reduced_matrix, axis=0)[1]
        min_x = np.amin(reduced_matrix, axis=0)[0]
        min_y = np.amin(reduced_matrix, axis=0)[1]
        plt.xlim((min_x * 1.2, max_x * 1.2))
        plt.ylim((min_y * 1.2, max_y * 1.2))
        for word_index in np.arange(args.voc_size):
            if word2vec.get_word_by_index(word_index) is not None:
                word_text = word2vec.get_word_by_index(word_index)
                if word2vec.get_word_count(word_text) >= 50:
                    plt.scatter(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], 0.5)
                    plt.text(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], word_text, fontsize=2.5)
                    """plt.annotate(word2vec.get_word_by_index(word_index),
                                 (reduced_matrix[word_index, 0], reduced_matrix[word_index, 1]),
                                 arrowprops=dict(facecolor='red', shrink=0.025))"""
                # end if
            # end if
        # end for

        # Save image
        plt.savefig(args.output)

        # Reset word prediction
        esn_word_prediction.reset()
    # end if

# end if
