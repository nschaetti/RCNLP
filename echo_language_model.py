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

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Echo Word Prediction Experience"
ex_instance = "Echo Language Model"

# Reservoir Properties
rc_leak_rate = 1.0  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 50  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.1


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

    # Word2Vec
    word2vec = Word2Vec(dim=20, mapper='dense', sparsity=0.3)

    # ESN for word prediction
    esn_word_prediction = EchoWordPrediction(word2vec=word2vec, size=rc_size, leaky_rate=rc_leak_rate,
                                             spectral_radius=rc_spectral_radius, input_scaling=rc_input_scaling,
                                             input_sparsity=rc_input_sparsity, w_sparsity=rc_w_sparsity)

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
        esn_word_prediction.add(u"The dog is beautiful.")
        esn_word_prediction.add(u"My dog is smart and beautiful.")
        esn_word_prediction.add(u"My cat is beautiful and smart.")
        esn_word_prediction.add(u"I think the dog is smart.")
        esn_word_prediction.add(u"I think the cat is beautiful.")
        esn_word_prediction.add(u"I think the cat is smart.")
        esn_word_prediction.add(u"I think the dog is beautiful.")
        esn_word_prediction.add(u"I think my dog is smart and beautiful.")
        esn_word_prediction.add(u"I think my cat is beautiful and smart.")
        esn_word_prediction.add(u"Is he smart?")
        esn_word_prediction.add(u"Is she smart?")
        esn_word_prediction.add(u"Is he beautiful?")
        esn_word_prediction.add(u"Is she beautiful?")

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
        predictions.append(esn_word_prediction.predict(u"The dog is beautiful."))
        predictions.append(esn_word_prediction.predict(u"My dog is smart and beautiful."))
        predictions.append(esn_word_prediction.predict(u"My cat is beautiful and smart."))
        predictions.append(esn_word_prediction.predict(u"I think the dog is smart."))
        predictions.append(esn_word_prediction.predict(u"I think the cat is beautiful."))
        predictions.append(esn_word_prediction.predict(u"I think the cat is smart."))
        predictions.append(esn_word_prediction.predict(u"I think the dog is beautiful."))
        predictions.append(esn_word_prediction.predict(u"I think my dog is smart and beautiful."))
        predictions.append(esn_word_prediction.predict(u"I think my cat is beautiful and smart."))
        predictions.append(esn_word_prediction.predict(u"Is he smart?"))
        predictions.append(esn_word_prediction.predict(u"Is she smart?"))
        predictions.append(esn_word_prediction.predict(u"Is he beautiful?"))
        predictions.append(esn_word_prediction.predict(u"Is she beautiful?"))

        # Predicted vectors
        pred_vectors = dict()

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

        # Average distance with last vectors
        last_distances = np.array([])

        # Reduce
        for word in pred_vectors.keys():
            if pred_vectors[word].ndim > 1:
                average_vector = np.average(pred_vectors[word], axis=0)
                d = distance(average_vector, word2vec[word])
                last_distances = np.append(last_distances, [d])
                word2vec[word] = average_vector
            # end if
        # end for

        # Print information
        """print_diff(word2vec, "smart", "beautiful")
        print_diff(word2vec, "he", "she")
        print_diff(word2vec, "he", "come")
        print_diff(word2vec, "smart", "think")
        print_diff(word2vec, "cat", "dog")
        print_diff(word2vec, "is", "will")"""
        distances = np.array([])
        for word in word2vec.words():
            print_diff(word2vec, "she", word)
            distances = np.append(distances, [distance(word2vec["she"], word2vec[word])])
        # end for
        #print(word2vec['he'])
        #print(word2vec['she'])
        print(np.average(distances))
        print(np.average(last_distances))

        # Wait
        answer = raw_input("Continue?")
        if answer == "n":
            cont = False
        # end if
        print("")

        esn_word_prediction.reset()
    # end while
# end if
