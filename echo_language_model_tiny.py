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
from sklearn.decomposition import PCA

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
    word2vec = Word2Vec(dim=8, mapper='dense', sparsity=0.3)

    # ESN for word prediction
    esn_word_prediction = EchoWordPrediction(word2vec=word2vec, size=rc_size, leaky_rate=rc_leak_rate,
                                             spectral_radius=rc_spectral_radius, input_scaling=rc_input_scaling,
                                             input_sparsity=rc_input_sparsity, w_sparsity=rc_w_sparsity)

    # Init vectors
    """word2vec[u'she']        = np.array([ 1, 0, 0, 0, 0, 0, 0, 0])
    word2vec[u'is']         = np.array([ 0, 1, 0, 0, 0, 0, 0, 0])
    word2vec[u'smart']      = np.array([ 0, 0, 1, 0, 0, 0, 0, 0])
    word2vec[u'beautiful']  = np.array([ 0, 0, 0, 1, 0, 0, 0, 0])
    word2vec[u'.']          = np.array([ 0, 0, 0, 0, 1, 0, 0, 0])
    word2vec[u'he']         = np.array([ 0, 0, 0, 0, 0, 1, 0, 0])
    word2vec[u'I']          = np.array([ 0, 0, 0, 0, 0, 0, 0, 1])
    word2vec[u'think']      = np.array([-1, 0, 0, 0, 0, 0, 0, 0])
    word2vec[u'will']       = np.array([ 0,-1, 0, 0, 0, 0, 0, 0])
    word2vec[u'come']       = np.array([ 0, 0,-1, 0, 0, 0, 0, 0])
    word2vec[u'and']        = np.array([ 0, 0, 0,-1, 0, 0, 0, 0])
    word2vec[u'the']        = np.array([ 0, 0, 0, 0,-1, 0, 0, 0])
    word2vec[u'cat']        = np.array([ 0, 0, 0, 0, 0,-1, 0, 0])
    word2vec[u'dog']        = np.array([ 0, 0, 0, 0, 0, 0,-1, 0])
    word2vec[u'tomorrow']   = np.array([ 0, 0, 0, 0, 0, 0, 0,-1])"""

    cont = True
    while cont:
        # Add text example
        esn_word_prediction.add(u"She is smart.")
        esn_word_prediction.add(u"He is beautiful.")
        esn_word_prediction.add(u"I think he will come.")
        esn_word_prediction.add(u"I think she will come.")
        esn_word_prediction.add(u"I think she is smart and beautiful.")
        esn_word_prediction.add(u"I think she is beautiful and smart.")
        esn_word_prediction.add(u"I think he is smart and beautiful.")
        esn_word_prediction.add(u"I think he is beautiful and smart.")
        esn_word_prediction.add(u"He will come tomorrow.")
        esn_word_prediction.add(u"She will come tomorrow.")
        esn_word_prediction.add(u"The dog is smart.")
        esn_word_prediction.add(u"The cat is beautiful.")
        esn_word_prediction.add(u"The cat is smart.")

        # Print initial vectors
        """print(u"He")
        print(word2vec[u"he"])
        print(u"She")
        print(word2vec[u"She"])
        print(u"is")
        print(word2vec[u'is'])
        print(u"was")
        print(word2vec[u'was'])
        print(u"Smart")
        print(word2vec[u'smart'])
        print(u"Beautiful")
        print(word2vec[u'beautiful'])
        print(u".")
        print(word2vec[u'.'])
        print(u"#######################################")"""

        # Train
        esn_word_prediction.train()

        # Predict
        predictions = list()
        predictions.append(esn_word_prediction.predict(u"She is smart."))
        predictions.append(esn_word_prediction.predict(u"He is beautiful."))
        predictions.append(esn_word_prediction.predict(u"I think he will come."))
        predictions.append(esn_word_prediction.predict(u"I think she will come."))
        predictions.append(esn_word_prediction.predict(u"I think she is smart and beautiful."))
        predictions.append(esn_word_prediction.predict(u"I think she is beautiful and smart."))
        predictions.append(esn_word_prediction.predict(u"I think he is smart and beautiful."))
        predictions.append(esn_word_prediction.predict(u"I think he is beautiful and smart."))
        predictions.append(esn_word_prediction.predict(u"He will come tomorrow."))
        predictions.append(esn_word_prediction.predict(u"She will come tomorrow."))
        predictions.append(esn_word_prediction.predict(u"The dog is smart."))
        predictions.append(esn_word_prediction.predict(u"The cat is beautiful."))
        predictions.append(esn_word_prediction.predict(u"The cat is smart."))

        # Predicted vectors
        pred_vectors = dict()
        average_vectors = dict()

        # For each prediction
        for pred in predictions:
            for word in pred.keys():
                if word not in pred_vectors.keys():
                    pred_vectors[word] = pred[word]
                else:
                    pred_vectors[word] = np.vstack((pred_vectors[word], pred[word]))
                # end if
            # end for
        # end for

        # Reduce
        for word in pred_vectors.keys():
            if pred_vectors[word].ndim > 1:
                average_vectors[word] = np.average(pred_vectors[word], axis=0)
            # end if
        # end for

        # Print resulting
        """print(u"He")
        print(prediction2[u"he"])
        print(u"She")
        print(prediction1[u"she"])
        print(u"is")
        print(prediction1[u'is'])
        print(u"is")
        print(prediction2[u'is'])
        print(u"was")
        print(prediction3[u'was'])
        print(u"was")
        print(prediction4[u'was'])
        print(u"Smart")
        print(prediction1[u'smart'])
        print(u"Beautiful")
        print(prediction2[u'beautiful'])
        print(u".")
        print(prediction1[u'.'])
        print(u".")
        print(prediction2[u'.'])"""

        # Update voc
        word2vec[u"she"] = predictions[0][u"she"]
        word2vec[u"he"] = predictions[1][u"he"]
        word2vec[u"is"] = predictions[0][u"is"]
        word2vec[u"smart"] = predictions[0][u"smart"]
        word2vec[u"beautiful"] = predictions[1][u"beautiful"]
        word2vec[u"."] = predictions[1][u"."]
        word2vec[u"I"] = predictions[2][u"i"]
        word2vec[u"think"] = predictions[3][u"think"]
        word2vec[u"and"] = predictions[4][u"and"]
        word2vec[u"will"] = predictions[8][u"will"]
        word2vec[u"come"] = predictions[3][u"come"]
        word2vec[u"the"] = predictions[10][u"the"]
        word2vec[u"dog"] = predictions[10][u"dog"]
        word2vec[u"cat"] = predictions[11][u"cat"]
        word2vec[u"tomorrow"] = predictions[9][u"tomorrow"]

        # Put all vectors in a matrix
        vectors_matrix = np.array([])
        for index, pred in enumerate(predictions):
            for word in pred.keys():
                if index == 0:
                    vectors_matrix = pred[word]
                else:
                    vectors_matrix = np.vstack((vectors_matrix, pred[word]))
                # end if
            # end for
        # end for

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
        """for word1 in word2vec.words():
            for word2 in word2vec.words():
                print_diff(word2vec, word1, word2)
            # end for
        # end for"""

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
        """print(u"He")
        print(word2vec[u"he"])
        print(u"She")
        print(word2vec[u"She"])
        print(u"is")
        print(word2vec[u'is'])
        print(u"was")
        print(word2vec[u'was'])
        print(u"Smart")
        print(word2vec[u'smart'])
        print(u"Beautiful")
        print(word2vec[u'beautiful'])
        print(u".")
        print(word2vec[u'.'])
        print(u"*************************************")"""

        answer = raw_input("Display? ").lower()
        if answer == "pca" or answer == "tsne":
            # TSNE
            if answer == "tsne":
                model = TSNE(n_components=2, random_state=0)
                #np.set_printoptions(suppress=True)
                #reduced_matrix = model.fit_transform(word2vec.get_matrix())
                reduced_matrix = model.fit_transform(vectors_matrix)
                """print(reduced_matrix.shape)
                print(reduced_matrix)"""
            # end if

            # PCA
            if answer == "pca":
                model = PCA(n_components=2)
                #reduced_matrix = pca.fit_transform(word2vec.get_matrix())
                reduced_matrix = model.fit_transform(vectors_matrix)
                """print(model.explained_variance_ratio_)
                print(reduced_matrix.shape)
                print(reduced_matrix)"""
            # end if

            # Show
            plt.figure(figsize=(200, 200), dpi=100)
            max_x = np.amax(reduced_matrix, axis=0)[0]
            max_y = np.amax(reduced_matrix, axis=0)[1]
            min_x = np.amin(reduced_matrix, axis=0)[0]
            min_y = np.amin(reduced_matrix, axis=0)[1]
            plt.xlim((min_x * 1.2, max_x * 1.2))
            plt.ylim((min_y * 1.2, max_y * 1.2))
            #plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 10)
            """for row_id in range(0, 15):
                target_word = word2vec.words()[row_id]
                x = reduced_matrix[row_id, 0]
                y = reduced_matrix[row_id, 1]
                print("{} = ({}, {})".format(target_word, x, y))
                plt.annotate(target_word, (x, y))
            # end for"""
            """for index, pred in enumerate(predictions):
                for word in pred.keys():
                    reducted_vector = model.transform(pred[word])
                    plt.scatter(reducted_vector[:, 0], reducted_vector[:, 1], 10)
                    plt.annotate(word, (reducted_vector[0, 0], reducted_vector[0, 1]))
                # end for
            # end for"""
            for word in average_vectors:
                reducted_vector = model.transform(average_vectors[word])
                plt.scatter(reducted_vector[0, 0], reducted_vector[0, 1], 10)
                plt.annotate(word, (reducted_vector[0, 0], reducted_vector[0, 1]),
                             arrowprops=dict(facecolor='red', shrink=0.025))
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
