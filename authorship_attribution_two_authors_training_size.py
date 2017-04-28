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
import matplotlib.pyplot as plt
import pickle
import numpy as np
import Oger
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.classifiers.RCNLPEchoWordClassifier import RCNLPEchoWordClassifier
from core.tools.RCNLPLogging import RCNLPLogging
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution"
ex_instance = "Two Authors Exploring Training Size"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.1

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship clustering with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--samples", type=int, help="Samples", default=20)
    parser.add_argument("--step", type=int, help="Step for training size value", default=5)
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

    # >> 1. Choose a text to symbol converter.
    if args.converter == "pos":
        converter = RCNLPPosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = RCNLPTagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = RCNLPFuncWordConverter(resize=args.in_components, pca_model=pca_model)
    else:
        converter = RCNLPWordVectorConverter(resize=args.in_components, pca_model=pca_model)
    # end if

    # >> 3. Array for results
    success_rate_avg = np.array([])
    success_rate_std = np.array([])

    # Training set sizes
    training_set_sizes = np.arange(1, 100, args.step)

    # For each training size
    for training_size in training_set_sizes:
        logging.save_results("Training size ", training_size, display=True)

        # Average success rate for this training size
        training_size_average_success_rate = np.array([])

        # >> 4. Try n time
        for s in range(0, args.samples):

            # >> 5. Prepare training and test set.
            training_set_indexes = np.arange(0, training_size, 1)
            test_set_indexes = np.arange(training_size, 100, 1)

            # >> 6. Create Echo Word Classifier
            classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                                 input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                                 spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

            # >> 7. Add authors examples
            author_indexes = np.random.choice(np.arange(1, 51, 1), 2, replace=False)
            for author_index, author_id in enumerate((author_indexes[0], author_indexes[1])):
                author_path = os.path.join(args.dataset, "total", str(author_id))
                for file_index in training_set_indexes:
                    classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), author_index)
                # end for
            # end for

            # >> 8. Train model
            classifier.train()

            # >> 9. Test model performance
            success = 0.0
            count = 0.0
            for author_index, author_id in enumerate((author_indexes[0], author_indexes[1])):
                author_path = os.path.join(args.dataset, "total", str(author_id))
                for file_index in test_set_indexes:
                    author_pred = classifier.pred(os.path.join(author_path, str(file_index) + ".txt"))
                    if author_pred == author_index:
                        success += 1.0
                    # end if
                    count += 1.0
                # end for
            # end for

            # >> 11. Save results
            training_size_average_success_rate = np.append(training_size_average_success_rate, [(success / count) * 100.0])

            # Delete variables
            del classifier
        # end for

        # >> 10. Log success
        logging.save_results("Success rate ", np.average(training_size_average_success_rate), display=True)

        # Save results
        success_rate_avg = np.append(success_rate_avg, np.average(training_size_average_success_rate))
        success_rate_std = np.append(success_rate_std, np.std(training_size_average_success_rate))

    # end for

    # Plot perfs
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", success rate vs training size.", x_label="Nb. text file",
                      y_label="Success rate", ylim=[-10, 120])
    plot.plot(y=success_rate_avg, x=training_set_sizes, yerr=success_rate_std, label="Success rate", subplot=1,
              marker='o', color='b')
    logging.save_plot(plot)

    # Open logging dir
    logging.open_dir()

# end if