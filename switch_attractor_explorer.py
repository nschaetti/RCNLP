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
ex_n_samples = 15

# Reservoir Properties
rc_leak_rate = 0.5  # Leak rate
rc_input_scaling = 0.1  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_threshold = 0.9  # Threshold to say that the input was already seen.
rc_sparsity = np.arange(0.05, 1.01, 0.05)   # Input sparsity
rc_w_sparsity = 1.0  # W Sparsity

# Data set properties
ds_data_set_size = 40  # Data set size (number of samples)
ds_memory_length = 140  # How long time to remember the entry
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
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance, exp_value="LR=0.5_size=100_slen=3000_mlen=140_IS=0.1_sparsity=0.05to1.0_2dim")
    logging.save_globals()

    # Symbols
    switching_symbol = [1, 0]
    #switch_back_symbol = [-1]
    other_symbols = [[0, 0], [0, 1]]

    # Save locals
    logging.save_variables(locals())

    # Parameter average results
    parameter_remembering_rates = []
    parameter_lucidity = []

    # Parameter error results
    parameter_remembering_rates_std = []
    parameter_lucidity_std = []

    # Generate the data set
    generator = RCNLPSwitchingAttractorLanguage(tag_symbol=switching_symbol, other_symbols=other_symbols,
                                                memory_length=ds_memory_length, sparsity=ds_sparsity)
    inputs, outputs = generator.generate_data_set(sample_length=ds_sample_length, n_samples=ds_data_set_size)

    # For each property value
    for sparsity in rc_sparsity:

        # Save performances
        sample_remembering_rate = []
        sample_lucidity = []

        # For each sample
        for s in range(ex_n_samples):
            # Nodes
            # reservoir = Oger.nodes.ReservoirNode(input_dim = 3, output_dim = rc_Size, input_scaling = rc_InputScaling,
            # set_initial_state =  False, my_initial_state = np.random.rand((rc_Size)))
            reservoir = Oger.nodes.LeakyReservoirNode(input_dim=len(switching_symbol), output_dim=rc_size,
                                                      input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                                      spectral_radius=rc_spectral_radius, sparsity=sparsity)
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

            # Save results for this sample
            sample_remembering_rate += [np.average(remembering_rate_set)]
            sample_lucidity += [np.average(lucidity_set)]

        # Average for this value
        parameter_remembering_rates += [np.average(sample_remembering_rate)]
        parameter_lucidity += [np.average(sample_lucidity)]

        # Error for this value
        parameter_remembering_rates_std += [np.std(sample_remembering_rate)]
        parameter_lucidity_std += [np.std(sample_lucidity)]

    # Log results
    logging.save_results("Remembering rates", parameter_remembering_rates, display=False)
    logging.save_results("Lucidity", parameter_lucidity, display=False)

    # Plot perfs
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)

    # First subplot
    plot.add_sub_plot(title=ex_instance + ", sparsity 0.05 to 1.0, 2 dim.", x_label="Inputs sparsity", y_label="Percentage", ylim=[-10, 120])
    plot.plot(y=parameter_remembering_rates, x=rc_sparsity, yerr=parameter_remembering_rates_std, label="Remembering rate", subplot=1, marker='o', color='b')
    plot.plot(y=parameter_lucidity, x=rc_sparsity, yerr=parameter_remembering_rates_std, label="Lucidity", subplot=1, marker='o', color='r')
    #plot.add_hline(value=100, length=1, subplot=1)
    #plot.add_hline(value=50, length=1, subplot=1)

    # Save plot
    logging.save_plot(plot)

    # Open logging dir
    logging.open_dir()

# end __main__
