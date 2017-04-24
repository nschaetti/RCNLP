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
import Oger
import numpy as np
from sklearn.decomposition import PCA
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator
import mdp
from scipy.cluster.vq import kmeans, vq
from skimage.draw import circle
from sklearn.metrics.cluster import v_measure_score

####################################################
# Function
####################################################


# Create a reservoir
def create_reservoir(input_dim, output_dim, input_scaling, leak_rate, spectral_radius, input_sparsity, w_sparsity):
    """
    Create a reservoir
    :param input_dim: Reservoir input dimension.
    :param output_dim: Reservoir size.
    :param input_scaling: Reservoir input scaling.
    :param leak_rate: Reservoir leaky rate.
    :param spectral_radius: Reservoir spectral radius.
    :param input_sparsity: Reservoir input sparsity.
    :param w_sparsity: Reservoir sparsity.
    :return: A MPD flow.
    """
    # Create the reservoir
    # reservoir = RCNLPWordReservoirNode(input_dim=n_symbols, output_dim=size, input_scaling=input_scaling,
    #                                   leak_rate=leak_rate, spectral_radius=spectral_radius,
    #                                   word_sparsity=word_sparsity, w_sparsity=w_sparsity)
    reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=output_dim, input_scaling=input_scaling,
                                              leak_rate=leak_rate, spectral_radius=spectral_radius,
                                              sparsity=input_sparsity, w_sparsity=w_sparsity)

    # Reset state at each call
    reservoir.reset_states = True

    # Create the flow
    r_flow = mdp.Flow([reservoir], verbose=1)

    return r_flow
# end create_reservoir


# Generate PCA image
def generate_pca_image(states1, states2, index1, index2, cs, size=256.0):
    """
    Generate PCA image.
    :param states1: State of first authors.
    :param states2: State of second authors.
    :param index1: Index of the first component to display.
    :param index2: Index of the second component to display.
    :param cs: Centroids.
    :param size: Size of the image.
    :return: The image as a numpy array.
    """
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
def save_pca_image(logging, reduced1, reduced2, index1, index2):
    """
    Save the PCA images.
    :param logging: Logging object.
    :param reduced1: Reduced states of first author.
    :param reduced2: Reduced states of second author.
    :param index1: Index of the first component.
    :param index2: Index of the second component.
    """
    # Total data
    data = np.vstack((reduced1[:, index1:index2+1], reduced2[:, index1:index2+1]))

    # Compute K-Means with K = 2
    centroids, _ = kmeans(data, 2)

    # Generate PCA image for 1th and 2th
    image = generate_pca_image(reduced1, reduced2, index1, index2, centroids)
    plot = RCNLPPlotGenerator(title="PCA Images", n_plots=1)
    plot.add_sub_plot(title="PCA", x_label="Principal component %d" % index1,
                      y_label="Principal component %d" % index2)
    plot.imshow(image)
    logging.save_plot(plot)
    del plot
# end save_pca_image


# Classify documents
def classify_documents(a_idx, a_index):
    """
    Classify all document of an author.
    :param a_idx: Classification decision for each states for an author.
    :param a_index: Document indexes of the author.
    :return: The classes for each document as a list.
    """
    # Classes
    classes = list()

    # Foreach documents
    for doc in a_index:
        # Classification of each representations
        idx = a_idx[doc[0]:doc[0]+doc[1]]

        # Majority vote
        vote = np.argmax(np.bincount(idx))

        # Add
        classes.append(vote)
    # end for

    return classes
# end classify_documents


# Get precision
def compute_precision(a_id, a_classes):
    """
    Get the precision measure.
    :param a_id: The author's ID.
    :param a_classes: The decision classes for each documents.
    :return: The average precision between 0 and 1.
    """
    return float(np.sum(np.array(a_classes) == a_id)) / float(len(a_classes)) * 100.0
# end compute_precision


# Get the V measure score.
def get_v_measure_score(a1_idx, a2_idx, a1_index, a2_index):
    """
    Get the average precision.
    :param a1_idx: Author 1 state classification decisions.
    :param a2_idx: Author 2 state classification decisions.
    :param a1_index: Author 1 indexes of documents.
    :param a2_index: Author 2 indexes of documents.
    :return: Average precision.
    """
    # Get author documents classification
    a1_classes = classify_documents(a1_idx, a1_index)
    a2_classes = classify_documents(a2_idx, a2_index)

    # Print resuts
    print(a1_classes)
    print(a2_classes)

    # State V measure score
    state_v_score = v_measure_score(np.hstack((np.repeat(0, len(a1_idx)), np.repeat(1, len(a2_idx)))),
                                    np.hstack((a1_idx, a2_idx)))

    # Document V measure score
    doc_v_score = v_measure_score(np.hstack((np.repeat(0, 100), np.repeat(1, 100))),
                                  np.hstack((a1_classes, a2_classes)))

    return state_v_score, doc_v_score
# end get_average_success_rate


# Generate reservoir states
def generate_reservoir_states(converter, the_flow, filename, remove_startup=0):
    """
    Generate reservoir states.
    :param converter: The text converter.
    :param the_flow: The MPD flow.
    :param filename: The text filename.
    :param remove_startup: How many startup states to remove.
    :return: The reservoir states.
    """
    # Convert the text to Temporal Vector Representation
    doc_array = converter(io.open(filename, 'r').read())

    # Generate the reservoir state
    states = the_flow(doc_array)[remove_startup:]

    return states
# end generate_reservoir_states


# Generate the states for all documents.
def generate_documents_states(converter, flow, text_directory, args):
    """
    Generate the states for all documents
    :param converter: The text converter.
    :param flow: The MDP flow.
    :param text_directory: Directory containing text files.
    :param args: Command parameters.
    :return: Reservoir states, document indexes, states count
    """
    # Documents indexes
    pos = 0
    doc_indexes = list()

    # Generate states for first author
    for index, text_file in enumerate(os.listdir(text_directory)):
        doc = generate_reservoir_states(converter, flow, os.path.join(text_directory, text_file), args.startup)
        if index == 0:
            doc_states = doc
        else:
            doc_states = np.vstack((doc_states, doc))
        # end if

        # Indexes
        doc_indexes.append((pos, doc.shape[0]))
        pos += doc.shape[0]
    # end for

    return doc_states, doc_indexes, doc_states.shape[0]
# end generate_documents_states


# Generate plot
def generate_plot(logging, ex_name, ex_instance, title, x_label, y_label, data, transpose=False, cmap=None):
    """

    :param logging: Logging object.
    :param ex_name: Experience name.
    :param ex_instance: Experience instance.
    :param title: Plot's title.
    :param x_label: X axis label.
    :param y_label: Y axis label.
    :param data: Data to display.
    :param transpose: Transpose the data.
    :param cmap: Color map.
    """
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", " + title, x_label=x_label, y_label=y_label)
    if transpose:
        plot.imshow(np.transpose(data), cmap=cmap)
    else:
        plot.imshow(data, cmap=cmap)
    # end if
    logging.save_plot(plot)
# end generate_plot


# Main function
def clustering_states(args, texts1, texts2, ex_name, ex_instance, size, input_scaling, leak_rate, spectral_radius,
                      input_sparsity, w_sparsity, logging, save_graph=False, pca_model=None, flow=None):

    # >> 1. Convert the text to symbolic or continuous representations
    if args.converter == "pos":
        converter = RCNLPPosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = RCNLPTagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = RCNLPFuncWordConverter(resize=args.in_components, pca_model=pca_model)
    else:
        converter = RCNLPWordVectorConverter(resize=args.in_components, pca_model=pca_model)
    # end if

    # >> 2. Create an echo state network
    if flow is None:
        flow = create_reservoir(converter.get_n_inputs(), size, input_scaling, leak_rate, spectral_radius,
                                input_sparsity, w_sparsity)
    # end if

    # >> 3. Generate Temporal Representations
    # Generate "temporal representations" for first author
    a1_states, a1_index, a1_n_samples = generate_documents_states(converter, flow, texts1, args)
    if save_graph:
        generate_plot(logging, ex_name, ex_instance, "Temporal representations for Author 1", "Time", "Neurons",
                      a1_states[:args.show_states], transpose=True, cmap='Greys')
    # end if

    # Generate "temporal representations" for second author
    a2_states, a2_index, a2_n_samples = generate_documents_states(converter, flow, texts2, args)
    if save_graph:
        generate_plot(logging, ex_name, ex_instance, "Temporal representations for Author 2", "Time", "Neurons",
                      a2_states[:args.show_states], transpose=True, cmap='Greys')
    # end if

    # >> 4. Complete states.
    complete_states = np.vstack((a1_states, a2_states))
    if save_graph:
        generate_plot(logging, ex_name, ex_instance, "Complete joined Reservoir states", "Time", "Neurons",
                      complete_states, transpose=True, cmap='Greys')
    # end if

    # Get average and std dev
    logging.save_results("Average neural activations", np.average(complete_states))
    logging.save_results("Std dev of neural activations", np.std(complete_states))

    # Same size for each authors in needed
    if args.homogene:
        if a1_states.shape[0] > a2_states.shape[0]:
            a1_states = a1_states[:a2_states.shape[0]]
        elif a2_states.shape[0] > a1_states.shape[0]:
            a2_states = a2_states[:a1_states.shape[0]]
        # end if
    # end if

    # >> 5. Join states.
    join_states = np.vstack((a1_states, a2_states))
    if save_graph:
        generate_plot(logging, ex_name, ex_instance, "Joined Reservoir states", "Time", "Neurons", join_states,
                      transpose=True, cmap='Greys')
    # end if

    # >> 6. Clustering
    if args.out_components != -1:
        # PCA
        pca = PCA(n_components=args.out_components)
        pca.fit(join_states)

        # Generate PCA image of principal components
        if args.pca_images and save_graph:
            # Generate PCA
            a1_states_pca = pca.transform(a1_states)
            a2_states_pca = pca.transform(a2_states)
            for c in np.arange(0, 8):
                save_pca_image(logging, a1_states_pca, a2_states_pca, c, c+1)
            # end for
        # end if

        # Reduce whole states
        join_states_reduced = pca.transform(join_states)

        # Get centroids for the whole components
        centroids, _ = kmeans(join_states_reduced, 2)

        # Assign each sample to a cluster
        idx, _ = vq(pca.transform(complete_states), centroids)
        a1_idx = idx[:a1_n_samples]
        a2_idx = idx[a1_n_samples:]
    else:
        # Get centroids for the whole components
        centroids, _ = kmeans(join_states, 2)

        # Assign each sample to a cluster
        idx, _ = vq(complete_states, centroids)
        a1_idx = idx[:a1_n_samples]
        a2_idx = idx[a1_n_samples:]
    # end if

    # Compute average precision
    return get_v_measure_score(a1_idx, a2_idx, a1_index, a2_index)
# end main
