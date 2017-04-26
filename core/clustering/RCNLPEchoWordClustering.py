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


class RCNLPEchoWordClustering(object):

    # Constructor
    def __init__(self, size, leak_rate, input_scaling, w_sparsity, input_sparsity, spectral_radius, converter):
        # Properties
        self._input_dim = converter.get_n_inputs() * 2
        self._output_dim = size
        self._leak_rate = leak_rate
        self._input_scaling = input_scaling
        self._w_sparsity = w_sparsity
        self._input_sparsity = input_sparsity
        self._spectral_radius = spectral_radius
        self._converter = converter
        self._same_examples = list()
        self._different_examples = list()

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
        self._flow = mdp.Flow([self._reservoir, self._readout], verbose=1)
    # end __init__

    # Generate data from text file
    def generate_data_from_text(self, text_file):
        return self._converter(io.open(text_file).read())
    # end generate_data_from_text

    # Add example for same authored texts.
    def add_same_author_example(self, text1, text2):
        self._same_examples.append((text1, text2))
    # end add_example

    # Add example for different authored texts.
    def add_different_author_example(self, text1, text2):
        self._different_examples.append((text1, text2))
    # end add_different_author_example

    # Generate data set
    def generate_data_set(self, examples, output):
        X = list()
        Y = list()

        # For each examples
        for example in examples:
            print("Creating examples %s and %s." % (example[0], example[1]))

            # Generate two data arrays
            x1 = self.generate_data_from_text(example[0])
            x2 = self.generate_data_from_text(example[1])

            # Same length
            if x1.shape[0] > x2.shape[0]:
                x1 = x1[:x2.shape[0]]
            elif x2.shape[0] > x1.shape[0]:
                x2 = x2[:x1.shape[0]]
            # end if

            # Stack
            x = np.hstack((x1, x2))

            # Output
            y = np.repeat(output, x1.shape[0], axis=0)
            y.shape = (x1.shape[0], 1)

            X.append(x)
            Y.append(y)
        # end for

        return X, Y
    # end generate_data_set

    # Train the model
    def train(self):
        # Inputs outputs
        X = list()
        Y = list()

        # For each same author examples
        print("Generating same author data set...")
        X_, Y_ = self.generate_data_set(self._same_examples, 1)
        X += X_
        Y += Y_

        # For each different author examples
        print("Generating different author data set...")
        X_, Y_ = self.generate_data_set(self._different_examples, 0)
        X += X_
        Y += Y_

        # Create data
        data = [None, zip(X, Y)]

        # Train the model
        print("Training model...")
        print(datetime.now().strftime("%H:%M:%S"))
        self._flow.train(data)
        print(datetime.now().strftime("%H:%M:%S"))
    # end train

    # Predict if two text files are from the same author.
    def pred(self, text1_file, text2_file):
        # Get reservoir inputs
        x1 = self._converter(io.open(text1_file).read())
        x2 = self._converter(io.open(text2_file).read())

        # Same length
        if x1.shape[0] > x2.shape[0]:
            x1 = x1[:x2.shape[0]]
        elif x2.shape[0] > x1.shape[0]:
            x2 = x2[:x1.shape[0]]
        # end if

        # Stack
        x = np.hstack((x1, x2))

        # Get reservoir response
        y = self._flow(x)

        y -= np.min(y)
        y /= np.max(y)
        plt.xlim([0, len(y[:, 0])])
        plt.ylim([0.0, 1.0])
        plt.plot(y[:, 0], color='r', label='Author 1')
        plt.plot(np.repeat(np.average(y[:, 0]), len(y[:, 0])), color='r', label='Author 1 average', linestyle='dashed')
        plt.show()

        # Classify
        if np.average(y[:, 0]) > 0.5:
            return True
        else:
            return False
        # end if
    # end pred

    # Predict if two texts are from the same author.
    def pred_text(self, text1, text2):
        pass
    # end pred_text

    # Get predictions probabilities from file
    def predictions_from_file(self, text1_file, text2_file):
        pass
    # end predictions_from_file

    # Get predictions probabilities from file
    def predictions_from_text(self, text1, text2):
        pass
    # end predictions_from_file

# end RCNLPEchoWordClassifier
