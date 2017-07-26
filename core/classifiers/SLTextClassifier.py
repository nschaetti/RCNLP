#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.SLTextClassifier.py
# Description : Statistical language text classifier.
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
from sys import getsizeof
import decimal
from .TextClassifier import TextClassifier


# Statistical language text classifier
class SLTextClassifier(TextClassifier):

    # Constructor
    def __init__(self, classes, smoothing, smoothing_param):
        """
        Constructor
        :param classes:
        :param smoothing:
        :param smoothing_param:
        """
        # Class super class
        super(SLTextClassifier, self).__init__(classes=classes)

        # Properties
        self._n_token = 0
        self._n_total_token = 0

        # Init dictionaries
        self._token_counters = dict()
        self._class_counters = dict()

        # Smoothing
        self._smoothing = smoothing
        self._smoothing_param = smoothing_param
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get token count
    def get_token_count(self):
        """
        Get token count
        :return:
        """
        return len(self._token_counters.keys())
    # end get_token_count

    # Train the model
    def train(self, x, y, verbose=False):
        """
        Train
        :param x: Example's inputs
        :param y: Example's outputs
        :param verbose: Verbosity
        """
        # Tokens
        tokens = spacy.load('en')(x)

        # For each token
        for token in tokens:
            token_text = token.text.lower()
            # Token counters
            try:
                self._token_counters[token_text] += 1.0
            except KeyError:
                self._token_counters[token_text] = 1.0
                self._n_token += 1.0
            # end try

            # Create entry in class counter
            try:
                probs = self._class_counters[token_text]
            except KeyError:
                self._class_counters[token_text] = dict()
            # end try

            # Class counters
            if y in self._class_counters[token_text].keys():
                self._class_counters[token_text][y] += 1.0
            else:
                self._class_counters[token_text][y] = 1.0
            # end if

            # One more token
            self._n_total_token += 1.0
        # end token
    # end train

    ##############################################
    # Override
    ##############################################

    # Get token probability
    def __getitem__(self, item):
        """
        Get token probability
        :param item:
        :return:
        """
        # Probs
        probs = dict()

        # Set default
        for c in self._classes:
            try:
                probs[c] = self._class_counters[item][c] / self._token_counters[item]
            except KeyError:
                probs[c] = 0.0
                # end try
        # end for

        return probs
    # end __getitem__

    # To String
    def __str__(self):
        """
        To string
        :return:
        """
        return "StatisticalModel(n_classes={}, n_tokens={}, mem_size={}o, " \
               "token_counters_mem_size={} Go, class_counters_mem_size={} Go, n_total_token={})" \
            .format(self._n_classes, self.get_token_count(),
                    getsizeof(self), round(getsizeof(self._token_counters) / 1073741824.0, 4),
                    round(getsizeof(self._class_counters) / 1073741824.0, 4), self._n_total_token)
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
        # Text's probabilities
        text_probs = dict()

        # Init
        for c in self._classes:
            text_probs[c] = decimal.Decimal(1.0)
        # end for

        # Parse text
        text_tokens = spacy.load('en')(x)

        # Get all tokens
        tokens = list()
        for token in text_tokens:
            tokens.append(token)
        # end for

        # For each token
        for token in tokens:
            token_text = token.text.lower()

            # Get token probs for each class
            try:
                token_probs = self[token_text]
                collection_prob = self._token_counters[token_text] / self._n_total_token
            except KeyError:
                continue
            # end try

            # For each class
            for c in self._classes:
                smoothed_value = SLTextClassifier.smooth(self._smoothing, token_probs[c], collection_prob,
                                                         len(tokens),
                                                         param=self._smoothing_param)
                text_probs[c] *= decimal.Decimal(smoothed_value)
            # end for
        # end for

        # Get highest prob
        max = decimal.Decimal(0.0)
        result_class = ""
        for c in self._classes:
            if text_probs[c] > max:
                max = text_probs[c]
                result_class = c
                # end if
        # end for

        return result_class, text_probs
    # end _classify

    # Reset the classifier
    def _reset_model(self):
        """
        Reset the classifier
        """
        # Properties
        self._n_token = 0
        self._n_total_token = 0

        # Init dictionaries
        self._token_counters = dict()
        self._class_counters = dict()
    # end reset

    ##############################################
    # Private
    ##############################################

    # Dirichlet prior smoothing function
    @staticmethod
    def smooth_dirichlet_prior(doc_prob, col_prob, doc_length, mu):
        """
        Dirichlet prior smoothing function
        :param doc_prob:
        :param col_prob:
        :param doc_length:
        :param mu:
        :return:
        """
        return (float(doc_length) / (float(doc_length) + float(mu))) * doc_prob + \
               (float(mu) / (float(mu) + float(doc_length))) * col_prob
    # end smooth

    # Jelinek Mercer smoothing function
    @staticmethod
    def smooth_jelinek_mercer(doc_prob, col_prob, param_lambda):
        """
        Jelinek Mercer smoothing function
        :param col_prob:
        :param param_lambda:
        :return:
        """
        return (1.0 - param_lambda) * doc_prob + param_lambda * col_prob
    # end smooth

    # Smoothing function
    @staticmethod
    def smooth(smooth_algo, doc_prob, col_prob, doc_length, param):
        """
        Smoothing function
        :param smooth_algo: Algo type
        :param doc_prob:
        :param col_prob:
        :param doc_length:
        :param param:
        :return:
        """
        if smooth_algo == "dp":
            return SLTextClassifier.smooth_dirichlet_prior(doc_prob, col_prob, doc_length, param)
        else:
            return SLTextClassifier.smooth_jelinek_mercer(doc_prob, col_prob, param)
        # end if
    # end smooth

# end SLTextClassifier
