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
from core.embeddings.Word2Vec import Word2Vec
from core.embeddings.EchoWordPrediction import EchoWordPrediction
from core.embeddings.WordPredictionDataset import WordPredictionDataset
from numpy import linalg as LA
from sklearn.manifold import TSNE
import pylab as plt

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Echo Word Prediction Experience"
ex_instance = "Echo Language Model Tiny"

# Reservoir Properties
rc_leak_rate = 1.0  # Leak rate
rc_input_scaling = 1.0  # Input scaling
rc_size = 50  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
rc_w_sparsity = 0.2
rc_input_sparsity = 0.5


def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))
# end distance


def print_diff(vectors, v1, v2):
    print("{} - {} : ".format(v1, v2))
    #print(vectors[v1])
    #print(vectors[v2])
    print(distance(vectors[v1], vectors[v2]))
    print("")
# end print_diff

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word prediction with Echo State Network")

    # Argument
    args = parser.parse_args()

    # Print precision
    #np.set_printoptions(precision=3)

    # Word2Vec
    word2vec = Word2Vec(dim=6, mapper='dense', sparsity=0.3)

    # ESN for word prediction
    esn_word_prediction = EchoWordPrediction(word2vec=word2vec, size=rc_size, leaky_rate=rc_leak_rate,
                                             spectral_radius=rc_spectral_radius, input_scaling=rc_input_scaling,
                                             input_sparsity=rc_input_sparsity, w_sparsity=rc_w_sparsity)

    # Init vectors
    word2vec[u'she'] = np.array([1, 0, 0, 0, 0, 0])
    word2vec[u'is'] = np.array([0, 1, 0, 0, 0, 0])
    word2vec[u'smart'] = np.array([0, 0, 1, 0, 0, 0])
    word2vec[u'beautiful'] = np.array([0, 0, 0, 1, 0, 0])
    word2vec[u'.'] = np.array([0, 0, 0, 0, 1, 0])
    word2vec[u'he'] = np.array([0, 0, 0, 0, 0, 1])

    cont = True
    while cont:
        # Add text example
        esn_word_prediction.add(u"She is smart.")
        esn_word_prediction.add(u"He is beautiful.")

        # Print initial vectors
        print(u"He")
        print(word2vec[u"he"])
        print(u"She")
        print(word2vec[u"She"])
        print(u"is")
        print(word2vec[u'is'])
        print(u"Smart")
        print(word2vec[u'smart'])
        print(u"Beautiful")
        print(word2vec[u'beautiful'])
        print(u".")
        print(word2vec[u'.'])
        print(u"#######################################")

        # Train
        esn_word_prediction.train()

        # Predict
        prediction1 = esn_word_prediction.predict(u"She is smart.")
        prediction2 = esn_word_prediction.predict(u"He is beautiful.")

        # Print resulting
        print(u"He")
        print(prediction2[u"he"])
        print(u"She")
        print(prediction1[u"she"])
        print(u"is")
        print(prediction1[u'is'])
        print(u"is")
        print(prediction2[u'is'])
        print(u"Smart")
        print(prediction1[u'smart'])
        print(u"Beautiful")
        print(prediction2[u'beautiful'])
        print(u".")
        print(prediction1[u'.'])
        print(u".")
        print(prediction2[u'.'])

        # Update voc
        word2vec[u"she"] = prediction1[u"she"]
        word2vec[u"he"] = prediction2[u"he"]
        word2vec[u"is"] = prediction1[u"is"]
        word2vec[u"smart"] = prediction1[u"smart"]
        word2vec[u"beautiful"] = prediction2[u"beautiful"]
        word2vec[u"."] = prediction2[u"."]

        # Print distance
        """distances = np.array([])
        for word in word2vec.words():
            print_diff(word2vec, u"she", word)
            distances = np.append(distances, [distance(word2vec[u"she"], word2vec[word])])
        # end for
        print(u"Average distance : {}".format(np.average(distances)))

        # Print distance
        distances = np.array([])
        for word in word2vec.words():
            print_diff(word2vec, u"smart", word)
            distances = np.append(distances, [distance(word2vec[u"smart"], word2vec[word])])
        # end for
        print(u"Average distance : {}".format(np.average(distances)))"""

        # Print distances
        for word1 in word2vec.words():
            for word2 in word2vec.words():
                print_diff(word2vec, word1, word2)
            # end for
        # end for

        # Length
        """lengths = np.array([])
        for word in word2vec.words():
            lengths = np.append(lengths, [LA.norm(word2vec[word])])
        # end for
        print(u"Average lengths : {}".format(np.average(lengths)))
        print(u"Max lengths : {}".format(np.max(lengths)))
        word2vec *= (1.0 / np.max(lengths))"""
        word2vec.normalize()

        # Print initial vectors
        print(u"He")
        print(word2vec[u"he"])
        print(u"She")
        print(word2vec[u"She"])
        print(u"is")
        print(word2vec[u'is'])
        print(u"Smart")
        print(word2vec[u'smart'])
        print(u"Beautiful")
        print(word2vec[u'beautiful'])
        print(u".")
        print(word2vec[u'.'])
        print(u"*************************************")

        answer = raw_input("Display? ").lower()
        if answer == "y":
            # TSNE
            model = TSNE(n_components=2, random_state=0)
            np.set_printoptions(suppress=True)
            reduced_matrix = model.fit_transform(word2vec.get_matrix())
            print(reduced_matrix.shape)
            print(reduced_matrix)

            # Show
            plt.figure(figsize=(200, 200), dpi=100)
            max_x = np.amax(reduced_matrix, axis=0)[0]
            max_y = np.amax(reduced_matrix, axis=0)[1]
            min_x = np.amin(reduced_matrix, axis=0)[0]
            min_y = np.amin(reduced_matrix, axis=0)[1]
            plt.xlim((min_x * 1.2, max_x * 1.2))
            plt.ylim((min_y * 1.2, max_y * 1.2))
            plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20)
            for row_id in range(0, 6):
                target_word = word2vec.words()[row_id]
                x = reduced_matrix[row_id, 0]
                y = reduced_matrix[row_id, 1]
                plt.annotate(target_word, (x, y))
            # end for
            plt.show()
        # end if

        # Continue
        answer = raw_input("Continue? ").lower()
        if answer == "n":
            cont = False
        # end if

        # Reset reservoir
        esn_word_prediction.reset()
    # end while

# end if
