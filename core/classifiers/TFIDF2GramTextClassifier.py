#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.TFIDFTextClassifier.py
# Description : TFIDF text classifier.
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

# Imports
import spacy
import math
from sys import getsizeof
import numpy as np
from numpy import linalg as LA
from .TextClassifier import TextClassifier


# TF-IDF text classifier
class TFIDF2GramTextClassifier(TextClassifier):
    """
    TF-IDF text classifier
    """

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes: Classes
        """
        # Class super class
        super(TFIDF2GramTextClassifier, self).__init__(classes=classes)

        # Properties
        self._n_tokens = 0.0
        self._n_total_tokens = 0.0
        self._classes_counts = dict()
        self._classes_token_count = dict()
        self._collection_counts = dict()
        self._classes_vectors = dict()
        self._classes_frequency = dict()
        self._token_position = dict()
        self._finalized = False

        # Class counters init
        for c in classes:
            self._classes_counts[c] = dict()
            self._classes_token_count[c] = 0.0
        # end for
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Train the model
    def train(self, x, y, verbose=False):
        """
        Train
        :param x: Example's inputs.
        :param y: Example's outputs.
        :param verbose: Verbosity
        """
        # Tokens
        tokens = spacy.load(self._lang)(x)

        # Preceding token
        preceding_token = None

        # For each token
        for token in tokens:
            # Filtering
            filtered, token_text = self._filter_token(token)
            token_text = token_text.lower()

            if filtered:
                if preceding_token is not None:
                    token_text = preceding_token.text.lower() + u" " + token_text
                    # Classes counts
                    try:
                        self._classes_counts[y][token_text] += 1.0
                    except KeyError:
                        self._classes_counts[y][token_text] = 1.0
                        self._n_tokens += 1.0
                    # end try

                    # Collection counts
                    try:
                        self._collection_counts[token_text] += 1.0
                    except KeyError:
                        self._collection_counts[token_text] = 1.0
                    # end try

                    # Classes token count
                    try:
                        self._classes_token_count[y] += 1.0
                    except KeyError:
                        self._classes_token_count[y] = 1.0
                    # end try

                    # Total tokens
                    self._n_total_tokens += 1.0
                # end if
                preceding_token = token
            # end if
        # end for
    # end train

    ##############################################
    # Override
    ##############################################

    # To String
    def __str__(self):
        """
        To string
        :return:
        """
        return "TFIDF2GramTextClassifier(n_classes={}, " \
               "n_tokens={}, mem_size={}o)".format(self._n_classes, self._n_tokens, getsizeof(self))
    # end __str__

    ##############################################
    # Private
    ##############################################

    # Classify a document
    def _classify(self, x):
        """
        Classify a document.
        :param x: Document's text.
        :return: A tuple with found class and values per classes.
        """
        # Tokens
        tokens = spacy.load(self._lang)(x)

        preceding_token = None
        d_vector = np.zeros(len(self._collection_counts.keys()), dtype='float64')
        for token in tokens:
            # Filtering
            filtered, token_text = self._filter_token(token)
            token_text = token_text.lower()

            if filtered:
                if preceding_token is not None:
                    try:
                        index = self._token_position[preceding_token.text.lower() + u" " + token_text]
                        d_vector[index] += 1.0
                    except KeyError:
                        pass
                    # end try
                # end if
                preceding_token = token
            # end if
        # end for

        # Normalize vector
        d_vector /= float(len(tokens))

        # For each classes
        similarity = np.zeros(len(self._classes_counts.keys()))
        for index, c in enumerate(self._classes_counts.keys()):
            similarity[index] = TFIDF2GramTextClassifier.cosinus_similarity(self._classes_vectors[c], d_vector)
        # end for

        return self._classes_counts.keys()[np.argmax(similarity)], similarity
    # end _classify

    # Finalize the training
    def _finalize_training(self, verbose=False):
        """
        Finalize the training
        """
        # Position of each token
        i = 0
        for token in sorted(self._collection_counts.keys()):
            self._token_position[token] = i
            i += 1
        # end for

        # Compute classes frequency
        for token in self._collection_counts.keys():
            count = 0.0
            for c in self._classes_counts.keys():
                try:
                    if self._classes_counts[c][token] > 0:
                        count += 1.0
                    # end if
                except KeyError:
                    pass
                # end try
            # end for
            self._classes_frequency[token] = count
            # end for
        # end if

        # For each classes
        for c in self._classes_counts.keys():
            c_vector = np.zeros(len(self._collection_counts.keys()), dtype='float64')
            for token in self._collection_counts.keys():
                index = self._token_position[token]
                try:
                    c_vector[index] = self._classes_counts[c][token]
                except KeyError:
                    c_vector[index] = 0
                # end try
            # end for
            c_vector /= float(self._classes_token_count[c])
            for token in self._collection_counts.keys():
                index = self._token_position[token]
                if self._classes_frequency[token] > 0:
                    c_vector[index] *= math.log(self._n_classes / self._classes_frequency[token])
                # end if
            # end for
            self._classes_vectors[c] = c_vector
        # end for
    # end _finalize_training

    # Reset the classifier
    def _reset_model(self):
        """
        Reset the classifier
        """
        # Properties
        self._n_tokens = 0.0
        self._n_total_tokens = 0.0
        self._classes_counts = dict()
        self._classes_token_count = dict()
        self._collection_counts = dict()
        self._classes_vectors = dict()
        self._classes_frequency = dict()
        self._token_position = dict()
        self._finalized = False

        # Class counters init
        for c in self._classes:
            self._classes_counts[c] = dict()
            self._classes_token_count[c] = 0.0
        # end for
    # end _reset_model

    ##############################################
    # Static
    ##############################################

    # Cosinus similarity
    @staticmethod
    def cosinus_similarity(a, b):
        """
        Cosinus similarity
        :param a:
        :param b:
        :return:
        """
        return np.dot(a, b) / (LA.norm(a) * LA.norm(b))
    # end cosinus_similarity

# end TFIDFTextClassifier
