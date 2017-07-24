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
import io
import numpy as np
import Oger
import mdp
from datetime import datetime
import matplotlib.pyplot as plt
from core.converters.RCNLPConverter import RCNLPConverter
from .TextClassifier import TextClassifier


# Echo Word classifier model
class EchoWordClassifier(TextClassifier):
    """
    Echo Word classifier model
    """

    # Variables
    _verbose = False

    # Constructor
    def __init__(self, classes, size, leak_rate, input_scaling, w_sparsity, input_sparsity, spectral_radius, converter,
                 w=None):
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

    # Reset learning but keep reservoir
    def reset(self):
        """
        Reset model learned parameters
        """
        del self._readout, self._flow

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=0)
    # end reset

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

    # Predict the class of a text
    def predict(self, text):
        """
        Predict class of a text file
        :param text: The text
        :return: Predicted class and classes probabilities
        """
        return self._classify(text)
    # end predict

    ##############################################
    # Override
    ##############################################

    ##############################################
    # Private
    ##############################################

    # Finalize the training phase
    def _finalize_training(self):
        """
        Finalize the training phase
        """
        # Inputs outputs
        X = list()
        Y = list()

        # For each training text file
        for text in self._examples.keys():
            x, y = self._generate_training_data(text, self._examples[text])
            X.append(x)
            Y.append(y)
        # end for

        # Create data
        data = [None, zip(X, Y)]

        # Pre-log
        if self._verbose:
            print("Training model...")
            print(datetime.now().strftime("%H:%M:%S"))
        # end if

        # Train the model
        self._flow.train(data)

        # Post-log
        if self._verbose:
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

        # Get maximum probability class
        return np.argmax(np.average(y, 0)), np.average(y, 0)
    # end _classify

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

        # Generate x and y
        return RCNLPConverter.generate_data_set_inputs(reps, self._n_classes, author)
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