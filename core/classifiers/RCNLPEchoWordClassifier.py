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
import math
import mdp
import matplotlib.pyplot as plt
from core.converters.RCNLPConverter import RCNLPConverter
from datetime import datetime


class RCNLPEchoWordClassifier(object):

    # Constructor
    def __init__(self, size, leak_rate, input_scaling, w_sparsity, input_sparsity, spectral_radius, converter,
                 n_classes):
        # Properties
        self._input_dim = converter.get_n_inputs()
        self._output_dim = size
        self._leak_rate = leak_rate
        self._input_scaling = input_scaling
        self._w_sparsity = w_sparsity
        self._input_sparsity = input_sparsity
        self._spectral_radius = spectral_radius
        self._converter = converter
        self._n_classes = n_classes
        self._examples = dict()

        # Create the reservoir
        self._reservoir = Oger.nodes.LeakyReservoirNode(input_dim=self._input_dim, output_dim=self._output_dim,
                                                        input_scaling=input_scaling,
                                                        leak_rate=leak_rate, spectral_radius=spectral_radius,
                                                        sparsity=input_sparsity, w_sparsity=w_sparsity)

        # Reset state at each call
        self._reservoir.reset_states = True

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=0)
    # end __init__

    # Generate training data from text file
    def generate_training_data_from_text(self, text_file, author):
        # Get Temporal Representations
        reps = self._converter(io.open(text_file).read())

        # Generate x and y
        return RCNLPConverter.generate_data_set_inputs(reps, self._n_classes, author)
    # end generate_training_data_from_text

    # Generate text data from text file
    def generate_test_data_from_text(self, text_file):
        return self._converter(io.open(text_file).read())
    # end generate_text_data_from_text

    # Add example
    def add_example(self, text_file, class_id):
        self._examples[text_file] = class_id
    # end add_example

    # Train the model
    def train(self):
        # Inputs outputs
        X = list()
        Y = list()

        # For each training text file
        #print("Generating data set...")
        for text_file in self._examples.keys():
            x, y = self.generate_training_data_from_text(text_file, self._examples[text_file])
            X.append(x)
            Y.append(y)
        # end for

        # Create data
        data = [None, zip(X, Y)]

        # Train the model
        #print("Training model...")
        #print(datetime.now().strftime("%H:%M:%S"))
        self._flow.train(data)
        #print(datetime.now().strftime("%H:%M:%S"))
    # end train

    # Predict the class of a text
    def pred(self, text_file, show_graph=False):
        # Get reservoir inputs
        x = self.generate_test_data_from_text(text_file)

        # Get reservoir response
        y = self._flow(x)
        y -= np.min(y)
        y /= np.max(y)
        # Plot results
        if show_graph:
            """y -= np.min(y)
            y /= np.max(y)"""
            plt.xlim([0, len(y[:, 0])])
            plt.ylim([0.0, 1.0])
            plt.plot(y[:, 0], color='r', label='Author 1')
            plt.plot(np.repeat(np.average(y[:, 0]), len(y[:, 0])), color='r', label='Author 1 average', linestyle='dashed')
            plt.plot(y[:, 1], color='b', label='Author 2')
            plt.plot(np.repeat(np.average(y[:, 1]), len(y[:, 1])), color='b', label='Author 2 average', linestyle='dashed')
            plt.plot(np.repeat(np.average(y), len(y[:, 1])), color='k', label='Average', linestyle='dashed')
            plt.show()
        # end if

        # Classify
        if np.average(y[:, 0]) > np.average(y[:, 1]):
            return 0, np.average(y[:, 0]), np.average(y[:, 1])
        else:
            return 1, np.average(y[:, 0]), np.average(y[:, 1])
        # end if
    # end pred

    # Predict the class of a string
    def pred_text(self, text):
        # Get reservoir inputs
        x = self._converter(text)

        # Get reservoir response
        y = self._flow(x)
        y -= np.min(y)
        y /= np.max(y)

        # Classify
        if np.average(y[:, 0]) > np.average(y[:, 1]):
            return 0, np.average(y[:, 0]), np.average(y[:, 1])
        else:
            return 1, np.average(y[:, 0]), np.average(y[:, 1])
        # end if
    # end pred_text

    # Get predictions probabilities from file
    def predictions_from_file(self, text_file):
        # Get reservoir inputs
        x = self.generate_test_data_from_text(text_file)

        # Get reservoir response
        y = self._flow(x)

        # Return probs
        return np.average(y[:, 0]), np.average(y[:, 1])
    # end predictions_from_file

    # Get predictions probabilities from file
    def predictions_from_text(self, text):
        # Get reservoir inputs
        x = self._converter(text)

        # Get reservoir response
        y = self._flow(x)

        # Return probs
        return np.average(y[:, 0]), np.average(y[:, 1])
    # end predictions_from_file

# end RCNLPEchoWordClassifier
