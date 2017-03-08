#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : already_seen.py
# Description : Here we try to evaluate the performances of reservoirs to say it it has already seen seen
# an input or not. We send a one in the reservoir and
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
from tools.already_seen_language import AlreadySeenLanguage
from tools.metrics import remembering_rate, equal_output, lucidity
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
# Experience settings
#
#########################################################################

# Reservoir Properties
rc_LeakRate					= 0.15												# Leak rate
rc_InputScaling 			= 0.25												# Input scaling
rc_Size 					= 50												# Reservoir size
rc_Sparsity					= 0                                                 # Reservoir sparsity
rc_SpectralRadius 			= 0.99												# Spectral radius

# Data set properties
rc_DatasetSize				= 40												# Dataset size (number of samples)
rc_MemoryLength				= 100                                               # How long time to remember the entry
rc_TestLength				= rc_DatasetSize - rc_TrainingLength				# Test set length (number of samples)
rc_Threshold				= 0.3                                               # Threshold to say that the input was already seen.
rc_TrainingLength			= 30												# Training set length (number of samples)
rc_SampleLength				= 1000                                              # Length of a sample
rc_SloppingMemory			= False                                             # Is the memory slowly fadding away?

####################################################
# Main function
####################################################
if __name__ == "__main__":

    # Generate the dataset
    generator = AlreadySeenLanguage([0,0,1], [[1,0,0],[0,1,0]], memory_length = rc_MemoryLength, slopping_memory = rc_SloppingMemory)
    inputs, outputs = generator.generateDataset(sample_length = rc_SampleLength, n_samples = rc_DatasetSize, sparsity = rc_Sparsity)

    # Reservoir
	#reservoir = Oger.nodes.ReservoirNode(input_dim = 3, output_dim = rc_Size, input_scaling = rc_InputScaling, set_initial_state =  False, my_initial_state = np.random.rand((rc_Size)))
	reservoir = Oger.nodes.LeakyReservoirNode(input_dim = 3, output_dim = rc_Size, input_scaling = rc_InputScaling, leak_rate = rc_LeakRate)
    readout = Oger.nodes.RidgeRegressionNode()

    # Training and test
    training_in, training_out	= inputs[0:rc_TrainingLength], outputs[0:rc_TrainingLength]
    test_in, test_out			= inputs[rc_TrainingLength:], outputs[rc_TrainingLength:]
	
    # Create the flow
    flow = mdp.Flow([reservoir, readout], verbose=1)

    # Reservoir input data
    data = [training_in, zip(training_in, training_out)]

    # Train
    flow.train(data)
	
    # For each test
    sample_pos = 0
    remembering_rates = 0.0
    lucidities = 0.0
    esn_outs = []
    final_outs = []
    for sample_test in test_in:

        # Evaluate
        esn_out = flow(sample_test)
        esn_outs += [esn_out]

        # To 1
        final_out = []
        for out in esn_out:
            if np.abs(out) > rc_Threshold:
                final_out = final_out + [1.0]
            else:
                final_out = final_out + [0.0]
        final_outs = final_outs + [final_out]

        remembering_rates += remembering_rate(final_out, test_out[sample_pos])
        lucidities += lucidity(final_out, test_out[sample_pos])

        sample_pos += 1
	
    print("Rembering rate : " + str(remembering_rates / float(len(test_in))))
    print("Lucidity : " + str(lucidities / float(len(test_in))))

    f, graph = plt.subplots(4)
    for p in range(0,4):
        average = np.average(esn_outs[sample_pos-1-p])
        a_average = np.full((len(esn_outs[sample_pos-1-p])), average)
        diff = np.abs(esn_outs[sample_pos-1-p] - a_average) / np.max(np.abs(esn_outs[sample_pos-1-p] - a_average))
        graph[p].plot(np.abs(esn_outs[sample_pos-1-p]))
        #graph[p].plot(diff)
        graph[p].plot(test_out[sample_pos-1-p])
        graph[p].plot(final_outs[sample_pos-1-p])
        #graph[p].plot(a_average)

    plt.show()
