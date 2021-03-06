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
import mdp
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.converters.LetterConverter import LetterConverter
from core.classifiers.RCNLPEchoWordClassifier import RCNLPEchoWordClassifier
from core.tools.RCNLPLogging import RCNLPLogging

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution Experience"
ex_instance = "Author Attribution"

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
    parser = argparse.ArgumentParser(description="RCNLP - Authorship attribution with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--author1", type=str, help="First author.", default="1")
    parser.add_argument("--author2", type=str, help="Second author.", default="2")
    parser.add_argument("--training-size", type=int, help="Training size.", default=4)
    parser.add_argument("--test-size", type=int, help="Test size.", default=40)
    parser.add_argument("--samples", type=int, help="Number of samples to use to assess accuracy.", default=20)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # >> 4. Generate W
    w = mdp.numx.random.choice([0.0, 1.0], (rc_size, rc_size), p=[1.0 - rc_w_sparsity, rc_w_sparsity])
    w[w == 1] = mdp.numx.random.rand(len(w[w == 1]))

    # Inputs
    reps = dict()
    reps['pos'] = [-1]
    reps['tag'] = [-1, 20]
    reps['fw'] = [-1, 60, 40, 20]
    reps['wv'] = [-1, 60, 40, 20]
    reps['letter'] = [-1]

    # For each representations
    for r in reps.keys():
        # For each size
        for in_size in reps[r]:
            print("For representation %s size %d" % (r, in_size))
            if in_size != -1:
                pca_model = pickle.load(open("models/pca_" + r + "_" + str(in_size) + ".p", 'r'))
            else:
                pca_model = None
            # end if

            # >> 1. Choose a text to symbol converter.
            if r == "pos":
                converter = RCNLPPosConverter(resize=-1, pca_model=pca_model)
            elif r == "tag":
                converter = RCNLPTagConverter(resize=-1, pca_model=pca_model)
            elif r == "fw":
                converter = RCNLPFuncWordConverter(resize=-1, pca_model=pca_model)
            elif r == "letter":
                converter = LetterConverter(resize=-1, pca_model=pca_model)
            else:
                converter = RCNLPWordVectorConverter(resize=-1, pca_model=pca_model)
            # end if

            # >> 3. Array for results
            average_success_rate = np.array([])

            # For each samples
            for s in range(0, args.samples):
                print("#")
                # >> 5. Prepare training and test set.
                training_set_indexes = np.arange(0, 100, 1)[s:s+args.training_size]
                test_set_indexes = np.delete(np.arange(0, 100, 1), training_set_indexes, axis=0)[:args.test_size]

                # >> 6. Create Echo Word Classifier
                classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                                     input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                                     spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity, w=w)

                # >> 7. Add examples
                for author_index, author_id in enumerate((args.author1, args.author2)):
                    author_path = os.path.join(args.dataset, "total", author_id)
                    for file_index in training_set_indexes:
                        classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), author_index)
                    # end for
                # end for

                # >> 8. Train model
                classifier.train()

                # >> 9. Test model performance
                success = 0.0
                count = 0.0
                for author_index, author_id in enumerate((args.author1, args.author2)):
                    author_path = os.path.join(args.dataset, "total", author_id)
                    for file_index in test_set_indexes:
                        author_pred, _, _ = classifier.pred(os.path.join(author_path, str(file_index) + ".txt"))
                        if author_pred == author_index:
                            success += 1.0
                        # end if
                        count += 1.0
                    # end for
                # end for

                # >> 11. Save results
                average_success_rate = np.append(average_success_rate, [(success / count) * 100.0])

                # Delete variables
                del classifier
            # end for

            # Log results
            logging.save_results("Average success rate ", np.average(average_success_rate), display=True)
            logging.save_results("Success rate std ", np.std(average_success_rate), display=True)
        # end for
    # end for

# end if