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
import mdp
from datetime import datetime
from sys import getsizeof
from .TextClassifier import TextClassifier
import matplotlib.pyplot as plt
from decimal import *


# Echo Word classifier model
class EchoWordClassifier(TextClassifier):
    """
    Echo Word classifier model
    """

    # Variables
    _verbose = False

    # Constructor
    def __init__(self, classes, size, leak_rate, input_scaling, w_sparsity, input_sparsity, spectral_radius, converter,
                 w=None, aggregation='average'):
        """
        Constructor
        :param classes: Set of possible classes
        :param size: Reservoir's size
        :param leak_rate: Reservoir's leaky rate
        :param input_scaling: Input scaling
        :param w_sparsity: Hidden layer sparsity
        :param input_sparsity: Input layer sparsity
        :param spectral_radius: Hidden layer matrix's spectral radius
        :param converter: Word to input converter
        :param w: Hidden layer matrix
        :param aggregation: Aggregation function (average, multiplication)
        """
        # Super
        super(EchoWordClassifier, self).__init__(classes=classes)

        # Properties
        self._input_dim = converter.get_n_inputs()
        self._output_dim = size
        self._leak_rate = leak_rate
        self._input_scaling = input_scaling
        self._w_sparsity = w_sparsity
        self._input_sparsity = input_sparsity
        self._spectral_radius = spectral_radius
        self._converter = converter
        self._examples = dict()
        self._last_y = []
        self._aggregation = aggregation

        # Create the reservoir
        self._reservoir = Oger.nodes.LeakyReservoirNode(input_dim=self._input_dim, output_dim=self._output_dim,
                                                        input_scaling=input_scaling,
                                                        leak_rate=leak_rate, spectral_radius=spectral_radius,
                                                        sparsity=input_sparsity, w_sparsity=w_sparsity, w=w)

        # Reset state at each call
        self._reservoir.reset_states = True

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=0)
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Train
    def train(self, x, y, verbose=False):
        """
        Add a training example
        :param x: Text file example
        :param y: Corresponding author
        :param verbose: Verbosity
        """
        self._examples[x] = y
        self._verbose = verbose
    # end train

    # Show debugging informations
    def debug(self):
        """
        Show debugging informations
        """
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        plt.xlim([0, len(self._last_y[:, 0])])
        plt.ylim([0.0, 1.0])
        for author in range(self._n_classes):
            plt.plot(self._last_y[:, author], color=colors[author], label=u"Author {}".format(author))
            plt.plot(np.repeat(np.average(self._last_y[:, author]), len(self._last_y[:, author])), color=colors[author],
                     label=u"Author {} average".format(author), linestyle=u"dashed")
        # end for
        plt.show()
    # end debug

    # Get debugging data
    def get_debugging_data(self):
        """
        Get debugging data
        :return: debugging data
        """
        return self._last_y
    # end _get_debugging_data

    ##############################################
    # Override
    ##############################################

    # To String
    def __str__(self):
        """
        To string
        :return:
        """
        return u"EchoWordClassifier(n_classes={}, size={}, spectral_radius={}, leaky_rate={}, mem_size={}o)".format(
            self._n_classes, self._output_dim, self._spectral_radius, self._leak_rate, getsizeof(self))
    # end __str__

    ##############################################
    # Private
    ##############################################

    # Finalize the training phase
    def _finalize_training(self, verbose=False):
        """
        Finalize the training phase
        :param verbose: Verbosity
        """
        # Inputs outputs
        X = list()
        Y = list()

        # For each training text file
        for index, text in enumerate(self._examples.keys()):
            if verbose:
                print(u"Training on {}/{}...".format(index, len(self._examples.keys())))
            # end if
            x, y = self._generate_training_data(text, self._examples[text])
            X.append(x)
            Y.append(y)
        # end for

        # Create data
        data = [None, zip(X, Y)]

        # Pre-log
        if verbose:
            print(u"Training model...")
            print(datetime.now().strftime("%H:%M:%S"))
        # end if

        # Train the model
        self._flow.train(data)

        # Post-log
        if verbose:
            print(datetime.now().strftime("%H:%M:%S"))
        # end if
    # end _finalize_training

    # Classify a text file
    def _classify(self, text):
        """
        Classify text
        :param text: Text to classify
        :return: Predicted class and class probabilities
        """
        # Get reservoir inputs
        x = self._generate_test_data(text)

        # Get reservoir response
        y = self._flow(x)
        y -= np.min(y)
        y /= np.max(y)

        # Save last y
        self._last_y = y

        # Get maximum probability class
        if self._aggregation == 'average':
            return np.argmax(np.average(y, 0)), np.average(y, 0)
        else:
            # Decimal score
            scores = list()
            for i in range(self._n_classes):
                scores[i] = Decimal(1.0)
            # end for

            # For each outputs
            for pos in range(y.shape[0]):
                for i in range(self._n_classes):
                    scores[i] = scores[i] * Decimal(y[pos, i])
                # end for
            # end for

            # Return the max
            max = 0.0
            for i in range(self._n_classes):
                if scores[i] > max:
                    max_c = i
                    max = scores[i]
                # end if
            # end for
            return max_c, scores
        # end if
    # end _classify

    # Reset learning but keep reservoir
    def _reset_model(self):
        """
        Reset model learned parameters
        """
        del self._readout, self._flow

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=0)

        # Examples
        self._examples = dict()
    # end _reset_model

    # Generate training data from text
    def _generate_training_data(self, text, author):
        """
        Generate training data from text file.
        :param text: Text
        :param author: Corresponding author.
        :return: Data set inputs
        """
        # Get Temporal Representations
        reps = self._converter(text)

        # Converter type
        converter_type = type(self._converter)

        # Generate x and y
        return converter_type.generate_data_set_inputs(reps, self._n_classes, author)
    # end generate_training_data

    # Generate text data from text file
    def _generate_test_data(self, text):
        """
        Generate text data from text file
        :param text: Text
        :return: Test data set inputs
        """
        return self._converter(text)
    # end generate_text_data

    ##############################################
    # Static
    ##############################################

# end RCNLPEchoWordClassifier
