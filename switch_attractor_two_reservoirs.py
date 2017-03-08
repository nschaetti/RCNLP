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
ex_instance = "Switch attractor, 2 reservoirs"

# Reservoir Properties
rc_leak_rate = 0.5 # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 50  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_threshold = 0.9  # Threshold to say that the input was already seen.

# Data set properties
ds_data_set_size = 40  # Data set size (number of samples)
ds_memory_length = 1000  # How long time to remember the entry
ds_training_length = 30  # Training set length (number of samples)
ds_test_length = ds_data_set_size - ds_training_length
ds_sample_length = 5000  # Length of a sample
ds_slopping_memory = False  # Is the memory slowly fading away?
ds_sparsity = 0  # Number of samples with no switching

####################################################
# Function
####################################################

# Create a reservoir
def create_reservoir(input_dim, output_dim, input_scaling, leak_rate, t_in, t_out):
    """

    :param input_dim:
    :param output_dim:
    :param input_scaling:
    :param leak_rate:
    :param t_in:
    :param t_out:
    :return:
    """

    r_reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=output_dim, input_scaling=input_scaling,
                                              leak_rate=leak_rate)
    r_readout = Oger.nodes.RidgeRegressionNode()

    # Create the flow
    r_flow = mdp.Flow([r_reservoir, r_readout], verbose=1)

    # Reservoir input data
    r_data = [t_in, zip(t_in, t_out)]

    # Train
    r_flow.train(r_data)

    return r_flow

# end create_reservoir

####################################################
# Main function
####################################################
if __name__ == "__main__":

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance, exp_value="LR=0.9vs0.05_slen=5000_mlen=1000_size=50")
    logging.save_globals()

    # Symbols
    switching_symbols = [1]
    other_symbols = [[0]]

    # Save locals
    logging.save_variables(locals())

    # Generate the data set
    generator = RCNLPSwitchingAttractorLanguage(tag_symbol=switching_symbols, other_symbols=other_symbols,
                                                memory_length=ds_memory_length, sparsity=ds_sparsity)
    inputs, outputs = generator.generate_data_set(sample_length=ds_sample_length, n_samples=ds_data_set_size)

    # Training and test
    training_in, training_out = inputs[0:ds_training_length], outputs[0:ds_training_length]
    test_in, test_out = inputs[ds_training_length:], outputs[ds_training_length:]

    # Reservoir 1
    flow1 = create_reservoir(input_dim=len(switching_symbols), output_dim=rc_size, input_scaling=rc_input_scaling,
                                  leak_rate=rc_leak_rate, t_in=training_in, t_out=training_out)

    # Reservoir 2
    flow2 = create_reservoir(input_dim=len(switching_symbols), output_dim=rc_size, input_scaling=rc_input_scaling,
                                  leak_rate=rc_leak_rate, t_in=training_in, t_out=training_out)

    # Measure sets
    remembering_rate_set1 = np.array([])
    lucidity_set1 = np.array([])
    remembering_rate_set2 = np.array([])
    lucidity_set2 = np.array([])

    # For each test
    predictions1 = []
    predictions2 = []

    # Final outputs
    final_outputs1 = []
    final_outputs2 = []

    # Sample position
    sample_pos = 0

    # For each test sample
    for sample_input in test_in:

        # Test output
        observed = test_out[sample_pos]

        # Evaluate
        predicted1 = flow1(sample_input)
        predicted2 = flow2(sample_input)

        # Add to system outputs
        predictions1 += [predicted1]
        predictions2 += [predicted2]

        # Measure performances 1
        remembering_rate1, final_output1 = RCNLPMetrics.remembering_rate(predicted1, observed, threshold=rc_threshold)
        remembering_rate_set1 = np.append(remembering_rate_set1, remembering_rate1)
        lucidity_set1 = np.append(lucidity_set1, RCNLPMetrics.lucidity(predicted1, observed, threshold=rc_threshold))

        # Measure performances 2
        remembering_rate2, final_output2 = RCNLPMetrics.remembering_rate(predicted2, observed, threshold=rc_threshold)
        remembering_rate_set2 = np.append(remembering_rate_set2, remembering_rate2)
        lucidity_set2 = np.append(lucidity_set2, RCNLPMetrics.lucidity(predicted2, observed, threshold=rc_threshold))

        # Add to final outputs
        final_outputs1 += [final_output1]
        final_outputs2 += [final_output2]

        # Next sample
        sample_pos += 1

    # endfor

    # Average performance
    logging.save_results("Remembering rate 1", np.average(remembering_rate_set1), display=True)
    logging.save_results("Lucidity 1", np.average(lucidity_set1), display=True)
    logging.save_results("Remembering rate 2", np.average(remembering_rate_set2), display=True)
    logging.save_results("Lucidity 2", np.average(lucidity_set2), display=True)

    # Plot results
    pos = 0
    for prediction1 in predictions1:

        # The observation
        observation = test_out[pos]

        # Prediction 2
        prediction2 = predictions2[pos]

        # The input
        system_input = test_in[pos]

        # Final  outputs
        final_output1 = final_outputs1[pos]
        final_output2 = final_outputs2[pos]

        # Plot pred and bos
        plot = RCNLPPlotGenerator(title=ex_name, n_plots=2)

        # First subplot
        plot.add_sub_plot(title=ex_instance + ", leak rate = 0.9", x_label="Steps", y_label="Values", ylim=[-1.2, 1.2])
        plot.plot(y=observation, label="Target signal", subplot=1, color='b')
        plot.plot(y=prediction1, label="ESN output", subplot=1, color='r')
        plot.add_hline(value=rc_threshold, length=ds_sample_length, subplot=1)
        plot.add_hline(value=-rc_threshold, length=ds_sample_length, subplot=1)

        # First subplot
        plot.add_sub_plot(title=ex_instance + ", leak rate = 0.05", x_label="Steps", y_label="Values", ylim=[-1.2, 1.2])
        plot.plot(y=observation, label="Target signal", subplot=2, color='b')
        plot.plot(y=prediction2, label="ESN output", subplot=2, color='r')
        plot.add_hline(value=rc_threshold, length=ds_sample_length, subplot=2)
        plot.add_hline(value=-rc_threshold, length=ds_sample_length, subplot=2)

        # Save plot
        logging.save_plot(plot)

        # Next test sample
        pos += 1
    # endfor

    # Open logging dir
    logging.open_dir()

# end __main__
