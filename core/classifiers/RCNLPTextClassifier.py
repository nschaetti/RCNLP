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

# Import packages
import numpy as np
import Oger
import math
import mdp


class RCNLPCharacterStreamClassifier(object):

    # Constructor
    def __init__(self, classes, size, input_scaling=-1, leak_rate = 0.95, input_sparsity=0.1, w_sparsity=0.1,
                 alphabet="abcdefghijklmnopqrstuvwxyz,.;:?!' ", use_uppercase=False, verbose=False):
        """
        Constructor.
        :param alphabet:
        :param use_uppercase:
        """
        # Properties
        self._classes = classes
        self._n_classes = len(classes)
        self._size = size
        self._leak_rate = leak_rate
        self._input_sparsity = input_sparsity
        self._w_sparsity = w_sparsity
        self._alphabet = alphabet
        self._use_uppercase = use_uppercase
        self._n_symbols = len(alphabet)
        if input_scaling == -1:
            self._input_scaling = math.floor(1.0 / self._n_symbols)
        else:
            self._input_scaling = input_scaling
        # end if
        self._symbols = dict()
        self._verbose = verbose
        self._generate_symbols()
        self._flow = self._create_esn()
    # end __init__

    # Train
    def train(self, docs, target, sep=' '):
        """
        Train the classifier.
        :param docs: List of documents (list of tokens)
        :param target: The true class of each document.
        """
        # Inputs/outputs
        inputs = np.array([])
        outputs = np.array([])

        # For reach document
        index = 0
        for doc in docs:
            doc_symbols = self._tokens_to_symbols(doc, sep=sep)
            np.append(inputs, doc_symbols)
            np.append(outputs, np.repeat(self._class_to_outputs(target[index]), len(doc)))
            index += 1
        # end for

        # Reservoir input data
        data = [inputs, zip(inputs, outputs), None]

        # Train
        self._flow.train(data)
    # end train

    # Classify by symbols
    def classify_symbols(self, tokens):
        """
        Classify a list of tokens
        :param tokens:
        :return:
        """

        # Tokens to symbols
        symbols = self._tokens_to_symbols(tokens, sep=' ')

        # Get prediction
        predictions = self._flow(symbols)

        # Count
        symbols_predictions = []
        for p in predictions:
            predicted_class = self._outputs_to_class(p)
            symbols_predictions += [predicted_class]
        # end for

        return symbols_predictions
    # end classify_symbols

    # Classify
    def classify(self, tokens):
        """
        Classify a document (list of tokens)
        :param tokens: List of tokens.
        :return: The predicted class.
        """

        # Tokens to symbols
        symbols = self._tokens_to_symbols(tokens, sep=' ')

        # Get prediction
        predictions = self._flow(symbols)

        # Classes counters
        counters = dict()
        for c in self._classes:
            counters[c] = 0
        # end for

        # Count
        for p in predictions:
            predicted_class = self._outputs_to_class(p)
            counters[predicted_class] += 1
        # end for

        maxi = 0
        max_class = ""
        for c in counters.keys():
            if counters[c] > maxi:
                maxi = counters[c]
                max_class = c
            # end if
        # end for

        return max_class
    # end classify

    # Class to outputs
    def _class_to_outputs(self, c):
        """
        Class to outputs
        :param c:
        :return:
        """
        outputs = np.zeros(self._n_classes)
        outputs[self._classes.index(c)] = 1.0
        return outputs
    # end _class_to_outputs

    # Outputs to classes
    def _outputs_to_class(self, outputs):
        """
        Outputs to class
        :param outputs:
        :return:
        """
        return self._classes[np.argmax(outputs)]
    # end _outputs_to_class

    # Create the Echo State Network
    def _create_esn(self):
        """
        Initialize the Echo State Network.
        """
        # Reservoir
        reservoir = Oger.nodes.LeakyReservoirNode(input_dim=self._n_symbols, output_dim=self._size,
                                                  input_scaling=self._input_scaling, leak_rate=self._leak_rate,
                                                  sparsity=self._input_sparsity, w_sparsity=self._w_sparsity)
        readout = Oger.nodes.RidgeRegressionNode()

        # Create the flow
        return mdp.Flow([reservoir, readout], verbose=self._verbose)
    # end _init_ESN

    # Generate discrete symbols
    def _generate_symbols(self):
        """
        Generate discrete symbols.
        :return:
        """
        # For each letter in alphabet
        index = 0
        for l in self._alphabet:
            symbol = np.zeros(self._n_symbols)
            symbol[index] = 1.0
            self._symbols[l] = symbol
            index += 1
        # end for
    # end _generate_symbols

    # Character to symbol
    def _character_symbols(self, c):
        """

        :param c:
        :return:
        """
        if self._use_uppercase:
            return self._symbols[c]
        else:
            return self._symbols[c.lower()]
        # end if
    # end _character_symbols

    # String to symbol list
    def _string_to_symbols(self, s):
        """

        :param s:
        :return:
        """
        symbols = []
        for l in s:
            symbols += [self._character_symbols(l)]
        # end for
        return symbols
    # end _string_to_symbols

    # Tokens to symbol list
    def _tokens_to_symbols(self, tokens, sep=' '):
        """

        :param tokens:
        :param sep:
        :return:
        """
        symbols = []
        for token in tokens:
            symbols += self._string_to_symbols(token)
            symbols += [self._character_symbols(sep)]
        # end for
        return symbols
    # end _tokens_to_symbols

# end RCNLPCharacterStreamClassifier
