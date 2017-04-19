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
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter
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
ex_name = "Author clustering Experience"
ex_instance = "Author clustering, PCA"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = (1.0 / 30.0)  # Input scaling
rc_size = 200  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_word_sparsity = 0.2
rc_w_sparsity = 0.1

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

    # Reset state at each call
    reservoir.reset_states = True

    # Create the flow
    r_flow = mdp.Flow([reservoir], verbose=1)

    return r_flow
# end create_reservoir


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


# Get average success rate
def get_average_precision(a1_idx, a2_idx, a1_index, a2_index):
    pass
# end get_average_success_rate


# Generate reservoir states
def generate_reservoir_states(converter, the_flow, filename, remove_startup=0):
    """
    Generate reservoir states.
    :param the_flow:
    :param filename:
    :param remove_startup:
    :return:
    """
    # Convert the text to Temporal Vector Representation
    doc_array = converter(io.open(filename, 'r').read())

    # Generate the reservoir state
    states = the_flow(doc_array)[remove_startup:]

    return states
# end generate_reservoir_states


# Generate the states for all documents.
def generate_documents_states(converter, flow, text_directory):
    """
    Generate the states for all documents
    :param flow:
    :param text_directory:
    :return:
    """
    # Documents indexes
    pos = 0
    doc_indexes = list()

    # Generate states for first author
    for index, text_file in enumerate(os.listdir(text_directory)):
        print("Generating state for author file %s (%d)" % (os.path.join(text_directory, text_file), index+1))
        if index == 0:
            doc = generate_reservoir_states(converter, flow, os.path.join(text_directory, text_file), args.startup)
            doc_states = doc
        else:
            doc = generate_reservoir_states(converter, flow, os.path.join(text_directory, text_file), args.startup)
            doc_states = np.vstack((doc_states, doc))
        # end if

        # Indexes
        doc_indexes.append((pos, doc.shape[0]))
        pos += doc.shape[0]
    # end for

    return doc_states, doc_indexes
# end generate_documents_states


# Generate plot
def generate_plot(title, x_label, y_label, data, transpose=False, cmap=None):
    """

    :param title:
    :param x_label:
    :param y_label:
    :param data:
    :param transpose:
    :param cmap:
    :return:
    """
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", " + title, x_label=x_label, y_label=y_label)
    if transpose:
        plot.imshow(np.transpose(data), cmap='Greys')
    else:
        plot.imshow(data, cmap='Greys')
    # end if
    logging.save_plot(plot)
# end generate_plot


# Main function
def main():

    # Convert the text to internal representations
    if args.converter == "pos":
        converter = RCNLPPosConverter(resize=args.in_components)
    elif args.converter == "tag":
        converter = RCNLPTagConverter(resize=args.in_components)
    elif args.converter == "fw":
        converter = RCNLPFuncWordConverter(resize=args.in_components)
    else:
        converter = RCNLPWordVectorConverter(resize=args.in_components)
    # end if

    # Create an echo state network
    flow = create_reservoir(converter.get_n_inputs(), rc_word_sparsity, rc_size, rc_input_scaling, rc_leak_rate,
                            rc_spectral_radius, rc_w_sparsity)

    # Generate "temporal representations" for first author
    a1_states, a1_index = generate_documents_states(converter, flow, args.author1)
    generate_plot("Temporal representations for Author 1", "Time", "Neurons", a1_states, transpose=True, cmap='Greys')

    # Generate "temporal representations" for second author
    a2_states, a2_index = generate_documents_states(converter, flow, args.author2)
    generate_plot("Temporal representations for Author 2", "Time", "Neurons", a2_states, transpose=True, cmap='Greys')

    # Same size for each authors in needed
    if args.homogene:
        if a1_states.shape[0] > a2_states.shape[0]:
            a1_states = a1_states[:a2_states.shape[0]]
        elif a2_states.shape[0] > a1_states.shape[0]:
            a2_states = a2_states[:a1_states.shape[0]]
        # end if
    # end if

    # Join states.
    join_states = np.vstack((a1_states, a2_states))
    generate_plot("Joined Reservoir states", "Time", "Neurons", a2_states, transpose=True, cmap='Greys')

    # PCA on all states.
    if args.out_components != -1:
        # PCA
        pca = PCA(n_components=args.out_components)
        pca.fit(join_states)

        # Generate PCA
        a1_states_pca = pca.transform(a1_states)
        a2_states_pca = pca.transform(a2_states)

        # Generate PCA image of principal components
        if args.pca_images:
            for c in np.arange(0, 8):
                save_pca_image(a1_states_pca, a2_states_pca, c, c+1)
            # end for
        # end if

        # Reduce whole states
        join_states_reduced = pca.transform(join_states)

        # Get centroids for the whole components
        centroids, _ = kmeans(join_states_reduced, 2)

        # Assign each sample to a cluster
        a1_idx, _ = vq(a1_states_pca, centroids)
        a2_idx, _ = vq(a2_states_pca, centroids)
    else:
        # Get centroids for the whole components
        centroids, _ = kmeans(join_states, 2)

        # Assign each sample to a cluster
        a1_idx, _ = vq(a1_states, centroids)
        a2_idx, _ = vq(a2_states, centroids)
    # end if

    # Compute average precision
    logging.save_results("Average precision", get_average_precision(a1_idx, a2_idx, a1_index, a2_index), display=True)

    # Open logging dir
    logging.open_dir()
# end main

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Author clustering with Part-Of-Speech to Echo State Network")

    # Argument
    parser.add_argument("--author1", type=str, help="First author text directory.")
    parser.add_argument("--author2", type=str, help="Second author text directory.")
    parser.add_argument("--startup", type=int, help="Number of start-up states to remove.", default=20)
    parser.add_argument("--out-components", type=int, help="Number of principal component to reduce reservoir states.", default=-1)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs.", default=-1)
    parser.add_argument("--homogene", action='store_true', help="Keep the same number of states for each authors.",
                        default=False)
    parser.add_argument("--pca-images", action='store_true', help="Generate image of principal components.",
                        default=False)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).")
    parser.add_argument("--lang", type=str, help="Language model", default='en')
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # Main
    main()

# end if