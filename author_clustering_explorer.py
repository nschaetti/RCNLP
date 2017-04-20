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

import argparse
from core.tools.RCNLPLogging import RCNLPLogging
import core.clustering.functions as cf
import numpy as np
import os
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator
from scipy.stats import ttest_1samp
import pickle

#########################################################################
#
# Experience settings
#
#########################################################################

# Exp. info
ex_name = "Author clustering Experience"
ex_instance = "Author clustering Explorer"

# Reservoir Properties
a_leak_rate = np.arange(0.05, 1.05, 0.1)
rc_leak_rate = "0.05-1.0"  # Leak rate
rc_input_scaling = 1.0  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.5

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Author clustering with Part-Of-Speech to Echo State Network")

    # Argument
    parser.add_argument("--texts", type=str, help="Text directory.")
    parser.add_argument("--startup", type=int, help="Number of start-up states to remove.", default=20)
    parser.add_argument("--out-components", type=int, help="Number of principal component to reduce reservoir states.",
                        default=-1)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs.", default=-1)
    parser.add_argument("--homogene", action='store_true', help="Keep the same number of states for each authors.",
                        default=False)
    parser.add_argument("--pca-images", action='store_true', help="Generate image of principal components.",
                        default=False)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).")
    parser.add_argument("--lang", type=str, help="Language model", default='en')
    parser.add_argument("--show-states", type=int, help="Number of states to show", default=500)
    parser.add_argument("--samples", type=int, help="Samples to estimate performances", default=20)
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default='')
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # PCA model
    pca_model = None
    if args.pca_model != "":
        pca_model = pickle.load(open(args.pca_model, 'r'))
    # end if

    # Results to analyze
    explore_results = np.array([])
    explore_deviation = np.array([])
    explore_t_test = np.array([])

    # Iterate
    for leak_rate in a_leak_rate:
        print("Evaluating performances for leak rate %f" % leak_rate)
        # Iterate
        results = np.array([])
        for i in np.arange(0, args.samples):
            authors_id = np.random.choice(49, 2, replace=False) + 1
            texts1 = os.path.join(args.texts, str(authors_id[0]))
            texts2 = os.path.join(args.texts, str(authors_id[1]))
            average_precision = cf.clustering_states(args=args, texts1=texts1, texts2=texts2, ex_name=ex_name,
                                                     ex_instance=ex_instance, input_scaling=rc_input_scaling,
                                                     input_sparsity=rc_input_sparsity, leak_rate=leak_rate,
                                                     logging=logging, size=rc_size, spectral_radius=rc_spectral_radius,
                                                     w_sparsity=rc_w_sparsity, save_graph=True if i == 0 else False,
                                                     pca_model=pca_model)
            logging.save_results("Precision 1 round " + str(i), average_precision[0], display=True)
            logging.save_results("Precision 2 round " + str(i), average_precision[1], display=True)
            results = np.append(results, average_precision[0])
            results = np.append(results, average_precision[1])
        # end for

        # Log
        logging.save_results("Overall average precision", np.average(results), display=True)
        logging.save_results("Overall std precision", np.std(results), display=True)
        logging.save_results("T-test", ttest_1samp(results, 50.0) * 100.0, display=True)

        # Save
        explore_results = np.append(explore_results, np.average(results))
        explore_deviation = np.append(explore_deviation, np.std(results))
        explore_t_test = np.append(explore_t_test, ttest_1samp(results, 50.0) * 100.0)
    # end for

    # First subplot
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title="Explorer", x_label="Leak rate", y_label="Precision", ylim=[0, 100], xlim=[0.0, 1.0])
    plot.plot(y=explore_results, x=a_leak_rate, yerr=explore_deviation, label="Precision", subplot=1, marker='o', color='b')
    plot.plot(y=explore_t_test, x=a_leak_rate, label="T-Test", subplot=1, marker='o', color='r')
    plot.add_hline(value=50, length=len(a_leak_rate), subplot=1)
    logging.save_plot(plot)

    # Open logging dir
    logging.open_dir()

# end if