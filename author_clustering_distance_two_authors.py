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
from core.converters.PosConverter import PosConverter
from core.converters.TagConverter import TagConverter
from core.converters.WVConverter import WVConverter
from core.converters.FuncWordConverter import FuncWordConverter
from core.classifiers.EchoWordClassifier import EchoWordClassifier
from core.tools.RCNLPLogging import RCNLPLogging

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Clustering Distance Two Authors"
ex_instance = "Author Clustering"

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
    parser.add_argument("--author1", type=int, help="Author 1's ID.")
    parser.add_argument("--author2", type=int, help="Author 2's ID.")
    parser.add_argument("--training-size", type=int, help="How many texts from the authors to use in the training")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
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
        converter = PosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = TagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = FuncWordConverter(resize=args.in_components, pca_model=pca_model)
    else:
        converter = WVConverter(resize=args.in_components, pca_model=pca_model)
    # end if

    # >> 2. Prepare training and test set indexes.
    n_training_samples = args.training_size
    n_test_samples = 100 - n_training_samples
    n_fold = int(100 / n_training_samples)
    indexes = np.arange(0, 100, 1)
    indexes.shape = (n_fold, n_training_samples)

    # >> 3. Array for results
    average_success_rate = np.array([])

    # >> 4. n-Fold cross validation
    for k in range(0, n_fold):
        print("%d-Fold" % k)

        # >> 5. Prepare training and test set.
        training_set_indexes = indexes[k]
        test_set_indexes = indexes
        test_set_indexes = np.delete(test_set_indexes, k, axis=0)
        test_set_indexes.shape = (100 - n_training_samples)

        # >> 6. Create Echo Word Classifier
        classifier = EchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                             input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                             spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

        # >> 7. Add authors examples
        print(training_set_indexes)
        for author_index, author_id in enumerate((args.author1, args.author2)):
            author_path = os.path.join(args.dataset, "total", str(args.author))
            for file_index in training_set_indexes:
                print("Adding positive example %s" % os.path.join(author_path, str(file_index) + ".txt"))
                classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), 0)
            # end for
        # end for

        # >> 7. Add negative examples
        n_negative_samples = 0
        author_index = 0
        text_index = 0
        while n_negative_samples < args.negative_samples:
            author_path = os.path.join(args.dataset, "total", str(negative_authors[author_index]))
            text_path = os.path.join(author_path, str(training_set_indexes[text_index]) + ".txt")
            print("Adding negative example %s" % text_path)
            classifier.add_example(text_path, 1)
            author_index += 1
            n_negative_samples += 1
            if author_index >= len(negative_authors):
                author_index = 0
                text_index += 1
                if text_index >= len(training_set_indexes):
                    break
                # end if
            # end if
        # end while

        # >> 8. Train model
        print("Training model...")
        classifier.train()

        # >> 9. Test model performance
        print("Testing model performances with text files from %s..." % os.path.join(args.dataset, "total"))
        print(test_set_indexes)
        success = 0.0
        count = 0.0
        # For each authors
        for author_id in np.arange(1, 51, 1):
            author_path = os.path.join(args.dataset, "total", str(author_id))
            print("Testing model performances with %d text files for author from %s..." % (test_set_indexes.shape[0],
                                                                                           author_path))
            test_count = 0
            for file_index in test_set_indexes:
                author_pred = classifier.pred(os.path.join(author_path, str(file_index) + ".txt"), True)
                if author_id == args.author and author_pred == 0:
                    success += 1.0
                elif author_id != args.author and author_pred == 1:
                    success += 1.0
                # end if
                count += 1.0
                test_count += 1
                if test_count >= args.test_size:
                    break
                # end if
            # end for
        # end for

        # >> 10. Log success
        logging.save_results("Number of file in test set ", count, display=True)
        logging.save_results("Number of success ", success, display=True)
        logging.save_results("Success rate ", (success / count) * 100.0, display=True)

        # >> 11. Save results
        average_success_rate = np.append(average_success_rate, [(success / count) * 100.0])

        # Delete variables
        del classifier
    # end for

    # Log results
    logging.save_results("Average success rate ", np.average(average_success_rate), display=True)
    logging.save_results("Success rate std ", np.std(average_success_rate), display=True)
    logging.save_results("Baseline ", float(len(negative_authors)) / 50.0 * 100.0, display=True)

# end if