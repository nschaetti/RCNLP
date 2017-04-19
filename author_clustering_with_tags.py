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

import io
import os
import argparse
import Oger
import numpy as np
from sklearn.decomposition import PCA
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.tools.RCNLPLogging import RCNLPLogging
from core.nodes.RCNLPWordReservoirNode import RCNLPWordReservoirNode
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator
import mdp
import matplotlib.pyplot as plt
import time
from scipy.cluster.vq import kmeans, vq
from skimage.draw import circle

#########################################################################
#
# Experience settings
#
#########################################################################

# Exp. info
ex_name = "Author clustering with Tags Experience"
ex_instance = "Author clustering, PCA"

# Reservoir Properties
rc_leak_rate = 0.3  # Leak rate
rc_input_scaling = 0.5  # Input scaling
rc_size = 800  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_word_sparsity = 1.0
rc_w_sparsity = 0.1

# Data set properties
ds_data_set_size = 40  # Data set size (number of samples)
ds_memory_length = 1200  # How long time to remember the entry
ds_training_length = 30  # Training set length (number of samples)
ds_test_length = ds_data_set_size - ds_training_length
ds_sample_length = 3000  # Length of a sample
ds_slopping_memory = False  # Is the memory slowly fading away?
ds_sparsity = 0  # Number of samples with no switching

####################################################
# Function
####################################################


# Create a reservoir
def create_reservoir(n_symbols, word_sparsity, size, input_scaling, leak_rate, spectral_radius, w_sparsity):
    """
    Create a reservoir.
    :param input_dim:
    :param output_dim:
    :param input_scaling:
    :param leak_rate:
    :param t_in:
    :param t_out:
    :return:
    """
    # Create the reservoir
    reservoir = RCNLPWordReservoirNode(input_dim=n_symbols, output_dim=size, input_scaling=input_scaling,
                                       leak_rate=leak_rate, spectral_radius=spectral_radius,
                                       word_sparsity=word_sparsity, w_sparsity=w_sparsity)

    # Create the flow
    r_flow = mdp.Flow([reservoir], verbose=1)

    return r_flow
# end create_reservoir


# Generate reservoir states
def generate_reservoir_states(the_flow, filename, remove_startup=0):
    # Convert the text to Temporal Vector Representation
    converter = RCNLPTagConverter()
    doc_array = converter(io.open(filename, 'r').read())

    # Generate the reservoir state
    states = the_flow(doc_array)[remove_startup:]

    return states
# end generate_reservoir_states


# Generate PCA image
def generate_pca_image(states1, states2, index1, index2, cs, size=256.0):
    n_samples = states1.shape[0] + states2.shape[0]
    n_ratio = 256.0 / n_samples

    # Min of each components
    min_axis1 = np.min([np.min(states1[:, index1]), np.min(states2[:, index1])])
    min_axis2 = np.min([np.min(states1[:, index2]), np.min(states2[:, index2])])

    # Max of each components
    max_axis1 = np.max([np.max(states1[:, index1]), np.max(states2[:, index1])])
    max_axis2 = np.max([np.max(states1[:, index2]), np.max(states2[:, index2])])

    # Range of each components
    range_axis1 = max_axis1 - min_axis1
    range_axis2 = max_axis2 - min_axis2

    # Multiplers
    axis1_mult = (size-1.0) / range_axis1
    axis2_mult = (size-1.0) / range_axis2

    # Create image
    im = np.zeros((int(size), int(size), 3))

    for s in states1:
        v1 = s[index1]
        v2 = s[index2]
        x = int((v1 - min_axis1) * axis1_mult)
        y = int((v2 - min_axis2) * axis2_mult)
        im[x, y, 1] += n_ratio
    # end for

    for s in states2:
        v1 = s[index1]
        v2 = s[index2]
        x = int((v1 - min_axis1) * axis1_mult)
        y = int((v2 - min_axis2) * axis2_mult)
        im[x, y, 2] += n_ratio
    # end for

    # Highest to 256
    im = (im / np.max(im)) * 255.0

    # Draw first centroids
    x = int((cs[0, 0] - min_axis1) * axis1_mult)
    y = int((cs[0, 1] - min_axis2) * axis2_mult)
    rr, cc = circle(x, y, 4)
    im[rr, cc] = (255, 0, 0)

    # Draw first centroids
    x = int((cs[1, 0] - min_axis1) * axis1_mult)
    y = int((cs[1, 1] - min_axis2) * axis2_mult)
    rr, cc = circle(x, y, 4)
    im[rr, cc] = (255, 0, 0)

    # Delete variables
    del min_axis1, min_axis2, max_axis1, max_axis2, range_axis1, range_axis2, axis1_mult, axis2_mult
    return im
# end generate_pca_image


# Save PCA image
def save_pca_image(reduced1, reduced2, index1, index2):
    # Total data
    data = np.vstack((reduced1[:, index1:index2+1], reduced2[:, index1:index2+1]))

    # Compute K-Means with K = 2
    centroids, _ = kmeans(data, 2)

    # Generate PCA image for 1th and 2th
    image = generate_pca_image(reduced1, reduced2, index1, index2, centroids)
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", PCA", x_label="Principal component %d" % index1,
                      y_label="Principal component %d" % index2)
    plot.imshow(image)
    logging.save_plot(plot)
    del plot
# end save_pca_image


# Average state success rate
def get_average_state_success_rate(idx, sample_size):
    class1 = np.argmax(np.bincount(idx[:sample_size]))
    if class1 == 0:
        class2 = 1
    else:
        class2 = 0
    count1 = np.sum(idx[:sample_size] == class1)
    count2 = np.sum(idx[sample_size:] == class2)
    return np.average([float(count1) / float(sample_size) * 100.0, float(count2) / float(sample_size) * 100.0])
# end get_average_state_success_rate


def main():
    # Create a reservoir
    flow = create_reservoir(45, rc_word_sparsity, rc_size, rc_input_scaling, rc_leak_rate,
                            rc_spectral_radius, rc_w_sparsity)

    # Generate states for first author
    for index, text_file in enumerate(os.listdir(args.author1)):
        print("Generating state for author 1 from file %s" % os.path.join(args.author1, text_file))
        if index == 0:
            state1 = generate_reservoir_states(flow, os.path.join(args.author1, text_file), args.startup)
        else:
            state1 = np.vstack(
                (state1, generate_reservoir_states(flow, os.path.join(args.author1, text_file), args.startup)))
        # end if
        if index == args.nfile:
            break
            # end if
    # end for

    # Display reservoir states for author 1
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", Reservoir states for Author 1", x_label="Time", y_label="Neurons")
    plot.imshow(np.transpose(state1), cmap='Greys')
    logging.save_plot(plot)
    print("Dimensions of states for author 1 : " + str(state1.shape))

    # Generate states for second author
    for index, text_file in enumerate(os.listdir(args.author2)):
        print("Generating state for author 2 from file %s" % os.path.join(args.author2, text_file))
        if index == 0:
            state2 = generate_reservoir_states(flow, os.path.join(args.author2, text_file), args.startup)
        else:
            state2 = np.vstack(
                (state2, generate_reservoir_states(flow, os.path.join(args.author2, text_file), args.startup)))
        # end if
        if index == args.nfile:
            break
            # end if
    # end for

    # Display reservoir states for author 2
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", Reservoir states for Author 2", x_label="Time", y_label="Neurons")
    plot.imshow(np.transpose(state2), cmap='Greys')
    logging.save_plot(plot)
    print("Dimensions of states for author 2 : " + str(state2.shape))

    # Same size for each authors
    if state1.shape[0] > state2.shape[0]:
        state1 = state1[:state2.shape[0]]
        sample_size = state1.shape[0]
    elif state2.shape[0] > state1.shape[0]:
        state2 = state2[:state1.shape[0]]
        sample_size = state2.shape[0]
    # end if

    # Join states
    join_states = np.vstack((state1, state2))
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", Joined Reservoir states", x_label="Time", y_label="Neurons")
    plot.imshow(np.transpose(join_states), cmap='Greys')
    logging.save_plot(plot)
    print("Dimensions of joined states : " + str(join_states.shape))

    # PCA
    pca = PCA(n_components=args.ncomponents)
    pca.fit(join_states)

    # Generate PCA
    reduced1 = pca.transform(state1)
    reduced2 = pca.transform(state2)
    treduced = pca.transform(join_states)

    # Generate PCA image for 1th and 2th
    for c in np.arange(0, 8):
        save_pca_image(reduced1, reduced2, c, c+1)
    # end for

    # Get centroids for the whole states and for components 1 and 2
    centroids, _ = kmeans(join_states, 2)
    rcentroids12, _ = kmeans(treduced[:, 0:2], 2)
    rcentroids15, _ = kmeans(treduced[:, 0:5], 2)

    # Assign each sample to a cluster
    idx, _ = vq(join_states, centroids)
    ridx12, _ = vq(treduced[:, 0:2], rcentroids12)
    ridx15, _ = vq(treduced[:, 0:5], rcentroids15)

    # Compute average state success rate
    logging.save_results("Average state ratio", get_average_state_success_rate(idx, sample_size), display=True)
    logging.save_results("PCA12 Average state ratio", get_average_state_success_rate(ridx12, sample_size), display=True)
    logging.save_results("PCA15 Average state ratio", get_average_state_success_rate(ridx15, sample_size), display=True)

    # Open logging dir
    logging.open_dir()
# end main

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Author clustering with Part-Of-Speech tags to Echo State Network")

    # Argument
    parser.add_argument("--author1", type=str, help="First author text directory")
    parser.add_argument("--author2", type=str, help="Second author text directory")
    parser.add_argument("--startup", type=int, help="Number of start-up states to remove")
    parser.add_argument("--ncomponents", type=int, help="Number of principal component to analyse")
    parser.add_argument("--nfile", type=int, help="Number of text files to analyze", default=-1)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # Main
    main()

# end if