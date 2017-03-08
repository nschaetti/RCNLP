#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : switch_attractor.py
# Description : Here we try to evaluate the performances of reservoirs to switch its outputs to one or zero for a
# long or infinite period of time when we put a one (one time) to its inputs.
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
from core.languages.RCNLPSwitchingAttractorLanguage import RCNLPSwitchingAttractorLanguage
from core.tools.RCNLPMetrics import RCNLPMetrics
from core.tools.RCNLPLogging import RCNLPLogging
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator
import mdp
import matplotlib.pyplot as plt
import numpy as np

#########################################################################
#
# Experience settings
#
#########################################################################

# Exp. info
ex_name = "Switch Attractor Experience"
ex_instance = "Switch attractor, symbol set"

# Reservoir Properties
rc_leak_rate = 0.05  # Leak rate
rc_input_scaling = 0.5  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
rc_threshold = 0.5  # Threshold to say that the input was already seen.
rc_sparsity = 0.2   # Input sparsity
rc_w_sparsity = 0.2  # W Sparsity

# Data set properties
ds_data_set_size = 40  # Data set size (number of samples)
ds_memory_length = 1200  # How long time to remember the entry
ds_training_length = 30  # Training set length (number of samples)
ds_test_length = ds_data_set_size - ds_training_length
ds_sample_length = 3000  # Length of a sample
ds_slopping_memory = False  # Is the memory slowly fading away?
ds_sparsity = 0  # Number of samples with no switching

####################################################
# Main function
####################################################
if __name__ == "__main__":

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance, exp_value="LR=0.05_size=100_SP=0.9_Thr=0.5_slen=3000_IS=0.5_mlen=1200_dim=1_sparsity=0.2_wsparsity=0.2")
    logging.save_globals()

    # Symbols
    switching_symbol = [1]
    #switching_symbol = [1, 0]
    #switching_symbol = [1, 0, 0]
    # switch_back_symbol = [-1]
    other_symbols = [[0]]
    #other_symbols = [[0, 0], [0, 1]]
    #other_symbols = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Save locals
    logging.save_variables(locals())

    # Generate the data set
    generator = RCNLPSwitchingAttractorLanguage(tag_symbol=switching_symbol, other_symbols=other_symbols,
                                                memory_length=ds_memory_length, sparsity=ds_sparsity)
    inputs, outputs = generator.generate_data_set(sample_length=ds_sample_length, n_samples=ds_data_set_size)

    # Nodes
    # reservoir = Oger.nodes.ReservoirNode(input_dim = 3, output_dim = rc_Size, input_scaling = rc_InputScaling,
    # set_initial_state =  False, my_initial_state = np.random.rand((rc_Size)))
    reservoir = Oger.nodes.LeakyReservoirNode(input_dim=len(switching_symbol), output_dim=rc_size,
                                              input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                              sparsity=rc_sparsity, w_sparsity=rc_w_sparsity)
    readout = Oger.nodes.RidgeRegressionNode()

    # Training and test
    training_in, training_out = inputs[0:ds_training_length], outputs[0:ds_training_length]
    test_in, test_out = inputs[ds_training_length:], outputs[ds_training_length:]

    # Create the flow
    flow = mdp.Flow([reservoir, readout], verbose=1)

    # Reservoir input data
    data = [training_in, zip(training_in, training_out)]

    # Train
    flow.train(data)

    # Measure sets
    remembering_rate_set = np.array([])
    lucidity_set = np.array([])

    # For each test
    predictions = []

    # Final outputs
    final_outputs = []

    # Sample position
    sample_pos = 0

    # For each test sample
    for sample_input in test_in:

        # Test output
        observed = test_out[sample_pos]

        # Evaluate
        predicted = flow(sample_input)

        # Add to system outputs
        predictions += [predicted]

        # Measure performances
        remembering_rate, final_output = RCNLPMetrics.remembering_rate(predicted, observed, threshold=rc_threshold)
        remembering_rate_set = np.append(remembering_rate_set, remembering_rate)
        lucidity_set = np.append(lucidity_set, RCNLPMetrics.lucidity(predicted, observed, threshold=rc_threshold))

        # Add to final outputs
        final_outputs += [final_output]

        # Next sample
        sample_pos += 1

    # endfor

    # Average performance
    logging.save_results("Remembering rate", np.average(remembering_rate_set), display=True)
    logging.save_results("Lucidity", np.average(lucidity_set), display=True)

    # Plot results
    pos = 0
    for prediction in predictions:

        # The observation
        observation = test_out[pos]

        # The input
        system_input = test_in[pos]

        # Final  outputs
        final_output = final_outputs[pos]

        # Plot pred and bos
        plot = RCNLPPlotGenerator(title=ex_name, n_plots=3)

        # First subplot
        plot.add_sub_plot(title=ex_instance + ", obs. vs pred.", x_label="Steps", y_label="Values", ylim=[-1.2, 1.2])
        plot.plot(y=observation, label="Target signal", subplot=1, color='b')
        plot.plot(y=prediction, label="ESN output", subplot=1, color='r')
        plot.add_hline(value=rc_threshold, length=ds_sample_length, subplot=1)
        plot.add_hline(value=-rc_threshold, length=ds_sample_length, subplot=1)

        # Second subplot
        plot.add_sub_plot(title=ex_instance + ", obs. vs f.pred.", x_label="Steps", y_label="Values", ylim=[-1.2, 1.2])
        plot.plot(y=observation, label="Target signal", subplot=2, color='b')
        plot.plot(y=final_output, label="Final output.", subplot=2, color='r')
        plot.add_hline(value=rc_threshold, length=ds_sample_length, subplot=2)
        plot.add_hline(value=-rc_threshold, length=ds_sample_length, subplot=2)

        # Third subplot
        plot.add_sub_plot(title=ex_instance + ", input.", x_label="Steps", y_label="Values", ylim=[-1.2, 1.2])
        plot.plot(y=system_input, label="System input", subplot=3, color='b')
        #plot.plot(y=system_input[:, 0], label="System input", subplot=3, color='b')
        #plot.plot(y=system_input[:, 1], label="System input", subplot=3, color='r')
        #plot.plot(y=system_input[:, 0], label="System input", subplot=3, color='b')
        #plot.plot(y=system_input[:, 1], label="System input", subplot=3, color='r')
        #plot.plot(y=system_input[:, 2], label="System input", subplot=3, color='g')
        plot.add_hline(value=rc_threshold, length=ds_sample_length, subplot=3)
        plot.add_hline(value=-rc_threshold, length=ds_sample_length, subplot=3)

        # Save plot
        logging.save_plot(plot)

        # Next test sample
        pos += 1
    # endfor

    # Open logging dir
    logging.open_dir()

# end __main__
