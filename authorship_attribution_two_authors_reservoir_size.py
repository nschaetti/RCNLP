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
import spacy
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
ex_instance = "Two Authors Exploring Reservoir Size"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.1

####################################################
# Functions
####################################################


def get_n_token(text_file):
    t_nlp = spacy.load(args.lang)
    doc = t_nlp(io.open(text_file, 'r').read())
    count = 0
    # For each token
    for index, word in enumerate(doc):
        count += 1
    # end for
    return count
# end get_n_token

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship clustering with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--author1", type=str, help="Author 1' ID.")
    parser.add_argument("--author2", type=str, help="Author 2's ID.")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--samples", type=int, help="Samples", default=20)
    parser.add_argument("--step", type=int, help="Step for reservoir size value", default=50)
    parser.add_argument("--min", type=int, help="Minimum reservoir size value", default=10)
    parser.add_argument("--max", type=int, help="Maximum reservoir size value", default=1000)
    parser.add_argument("--training-size", type=int, help="Training size", default=50)
    parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?", default=False)
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

    # Reservoir sizes
    reservoir_sizes = np.arange(args.min, args.max+1, args.step)

    # For each reservoir size
    for reservoir_size in reservoir_sizes:
        print("Reservoir size %d" % reservoir_size)

        # Average success rate for this training size
        reservoir_size_average_success_rate = np.array([])

        # >> 4. Try n time
        for s in range(0, args.samples):
            try:
                # >> 5. Prepare training and test set.
                training_set_indexes = np.arange(0, args.training_size, 1)
                test_set_indexes = np.arange(args.training_size, 100, 1)

                # >> 6. Create Echo Word Classifier
                classifier = RCNLPEchoWordClassifier(size=reservoir_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                                     input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                                     spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

                # >> 7. Add authors examples
                for author_index, author_id in enumerate((args.author1, args.author2)):
                    author_path = os.path.join(args.dataset, "total", str(author_id))
                    for file_index in training_set_indexes:
                        file_path = os.path.join(author_path, str(file_index) + ".txt")
                        classifier.add_example(file_path, author_index)
                    # end for
                # end for

                # >> 8. Train model
                classifier.train()

                # >> 9. Test model performance
                success = 0.0
                count = 0.0
                for author_index, author_id in enumerate((args.author1, args.author2)):
                    author_path = os.path.join(args.dataset, "total", str(author_id))
                    for file_index in test_set_indexes:
                        file_path = os.path.join(author_path, str(file_index) + ".txt")

                        # Success rate
                        if not args.sentence:
                            author_pred, _, _ = classifier.pred(os.path.join(author_path, str(file_index) + ".txt"))
                            if author_pred == author_index:
                                success += 1.0
                            # end if
                            count += 1.0
                        else:
                            # Sentence success rate
                            nlp = spacy.load(args.lang)
                            doc = nlp(io.open(file_path, 'r').read())
                            for sentence in doc.sents:
                                sentence_pred, _, _ = classifier.pred_text(sentence.text)
                                if sentence_pred == author_index:
                                    success += 1.0
                                # end if
                                count += 1.0
                            # end for
                        # end if
                    # end for
                # end for

                # >> 11. Save results
                print((success / count) * 100.0)
                reservoir_size_average_success_rate = np.append(reservoir_size_average_success_rate,
                                                               [(success / count) * 100.0])

                # Delete variables
                del classifier
            except:
                pass
            # end try
        # end for

        # >> 10. Log success
        logging.save_results("Reservoir size ", reservoir_size, display=True)
        logging.save_results("Success rate ", np.average(reservoir_size_average_success_rate), display=True)
        logging.save_results("Success rate std ", np.std(reservoir_size_average_success_rate), display=True)

        # Save results
        success_rate_avg = np.append(success_rate_avg, np.average(reservoir_size_average_success_rate))
        success_rate_std = np.append(success_rate_std, np.std(reservoir_size_average_success_rate))
    # end for

    for index, success_rate in enumerate(success_rate_avg):
        print("(%d, %f)" % (reservoir_sizes[index], success_rate))
    # end for

    for index, success_rate in enumerate(success_rate_std):
        print("(%d, %f)" % (reservoir_sizes[index], success_rate))
    # end for

    # Plot perfs
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", success rates vs training size.", x_label="Nb. tokens",
                      y_label="Success rates", ylim=[-10, 120])
    plot.plot(y=success_rate_avg, x=reservoir_sizes, yerr=success_rate_std, label="Success rate", subplot=1,
              marker='o', color='b')
    logging.save_plot(plot)

    # Open logging dir
    logging.open_dir()

# end if