#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
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
#

import Oger
from tools.multiple_memory_generator import MultipleMemoryGenerator
from tools.metrics import remembering_rate, equal_output, lucidity, MSSR, average_distance
from tools.word_reservoir_node import WordReservoirNode
import mdp
import os
import cPickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np

#########################################################################
#
# Experience properties
#
#########################################################################

# Reservoir properties
rc_SpectralRadius = 0.9  # Spectral radius
rc_Size = 100  # Taille du réservoir
# rc_Size                     = np.arange(100,1000,100)  # Taille du réservoir
# rc_InputScaling             = 1.0 / rc_NbDim  # Dimensionnement des entrées
rc_InputScaling = 0.5  # Dimensionnement des entrées
rc_LeakRate = 0.1  # Leak rate
rc_WSparsity = 0.1
rc_WordSparsity = 0.1
# rc_WordSparsity				= np.arange(0.1,1.0,0.1)

# Data set properties
rc_DatasetSize = 2  # Longueur du jeu de données
rc_TrainingLength = 1  # Longueur d'entrainement
rc_TestLength = rc_DatasetSize - rc_TrainingLength  # Longeur de test
# rc_MemoryShiffting          = 5
rc_MemoryShiffting = np.arange(1, 20, 1)
rc_SampleLength = 5000
rc_NbDim = 10
rc_NbSamples = 10

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":

    # File destination
    if len(sys.argv) >= 2:
        file_output = sys.argv[1]
    else:
        file_output = ""

    # Generate the dataset
    generator = MultipleMemoryGenerator()

    # Table of results for each values
    distances = []

    # Varying a property
    for shift in rc_MemoryShiffting:

        # Multiple sample
        samples = []
        for n in np.arange(0, rc_NbSamples):
            # Reservoir
            # reservoir = Oger.nodes.LeakyReservoirNode(input_dim = rc_NbDim, output_dim = rc_Size, input_scaling = rc_InputScaling, leak_rate = rc_LeakRate)
            reservoir = WordReservoirNode(input_dim=rc_NbDim, output_dim=rc_Size, input_scaling=rc_InputScaling,
                                          leak_rate=rc_LeakRate, w_sparsity=rc_WSparsity, word_sparsity=rc_WordSparsity)
            readout = Oger.nodes.RidgeRegressionNode()

            # Create the flow
            flow = mdp.Flow([reservoir, readout], verbose=1)

            # Generating data to learn
            inputs, outputs = generator.generateDataset(n_dim=rc_NbDim, memory_shiffting=shift,
                                                        sample_length=rc_SampleLength, n_samples=rc_DatasetSize)

            # Training and test
            training_in, training_out = inputs[0:rc_TrainingLength], outputs[0:rc_TrainingLength]
            test_in, test_out = inputs[rc_TrainingLength:], outputs[rc_TrainingLength:]

            # Reservoir input data
            data = [training_in, zip(training_in, training_out)]

            # Train
            flow.train(data)

            # Target output
            target = test_out[0]

            # Get result
            reservoir_output = flow(test_in[0])

            # Compare
            # print("Average distance : " + str(average_distance(target, reservoir_output)))
            samples += [average_distance(target, reservoir_output)]

        distances += [samples]
        # print("For size " + str(size) + " : " + str(np.average(sample)))

    # Put into average and std
    distance_averages = []
    distance_errors = []
    for dist in distances:
        distance_averages += [np.average(dist)]
        distance_errors += [np.std(dist)]

    # Plot
    plt.errorbar(rc_MemoryShiffting, distance_averages, yerr=distance_errors, fmt='-o')
    plt.show()

    #  Save to file
    if file_output != "":
        f = open(file_output, 'w')
        f.write("Averages : ")
        for a in distance_averages:
            f.write(str(a) + ",")
        f.write("\n")
        f.write("Errors : ")
        for e in distance_errors:
            f.write(str(e) + ",")
        f.close()
