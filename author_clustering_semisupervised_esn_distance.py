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
    parser.add_argument("--training-size", type=int, help="Number of texts from the author", default=1)
    parser.add_argument("--test-size", type=int, help="Number of texts to assess the model.", default=20)
    parser.add_argument("--negatives", type=int, help="Number of negative texts to use", default=1)
    parser.add_argument("--samples", type=int, help="Number of samples to use to assess accuracy.", default=20)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--threshold", type=float, help="Confidence threshold", default=0.5)
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
    success_rates = np.array([])
    same_probs = np.array([])
    diff_probs = np.array([])

    # >> 4. Try n time
    for s in range(args.samples):
        the_author = np.random.choice(np.arange(1, 51, 1))
        max_prob = 0.0
        max_cat = ''

        # >> 5. Prepare training and test set.
        training_set_indexes = np.arange(s, s+args.training_size, 1)
        test_set_indexes = np.delete(np.arange(0, 100, 1), training_set_indexes)[:args.test_size]
        negatives_set_indexes = np.arange(0, args.negatives, 1)
        other_authors = np.delete(np.arange(1, 51, 1), the_author-1)

        # >> 6. Create Echo Word Classifier
        classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                             input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                             spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

        # >> 7. Add authors examples
        author_path = os.path.join(args.dataset, "total", str(the_author))
        for file_index in training_set_indexes:
            file_path = os.path.join(author_path, str(file_index) + ".txt")
            classifier.add_example(file_path, 0)
        # end for

        # >> 8. Add negative examples
        others_path = os.path.join(args.dataset, "total", "others")
        for file_index in negatives_set_indexes:
            file_path = os.path.join(others_path, str(file_index) + ".txt")
            classifier.add_example(file_path, 1)
        # end for

        # >> 8. Train model
        classifier.train()

        # >> 9. Test model performances
        success = 0.0
        count = 0.0

        # >> 10. Test same author
        for file_index in test_set_indexes:
            file_path = os.path.join(author_path, str(file_index) + ".txt")

            # Doc. success rate
            if not args.sentence:
                author_pred, same_prob, diff_prob = classifier.pred(file_path)
                same_probs = np.append(same_probs, same_prob)
                if same_prob > max_prob:
                    max_prob = same_prob
                    max_cat = 'same'
                # end if
                if same_prob > diff_prob and same_prob > args.threshold:
                    success += 1.0
                # end if
                count += 1.0
            else:
                # Sentence success rate
                nlp = spacy.load(args.lang)
                doc = nlp(io.open(file_path, 'r').read())
                for sentence in doc.sents:
                    author_pred, same_prob, diff_prob = classifier.pred_text(sentence.text)
                    if same_prob > diff_prob and same_prob > args.threshold:
                        success += 1.0
                    # end if
                    count += 1.0
                # end for
            # end if
        # end for

        # >> 10. Test different author
        for file_index in test_set_indexes:
            other_author_path = os.path.join(args.dataset, "total", str(np.random.choice(other_authors)))
            file_path = os.path.join(other_author_path, str(file_index) + ".txt")

            if not args.sentence:
                author_pred, same_prob, diff_prob = classifier.pred(file_path)
                diff_probs = np.append(diff_probs, same_prob)
                if same_prob > max_prob:
                    max_prob = same_prob
                    max_cat = 'diff'
                # end if
                if same_prob > diff_prob and same_prob > args.threshold:
                    pass
                else:
                    success += 1.0
                # end if
                count += 1.0
            else:
                # Sentence success rate
                nlp = spacy.load(args.lang)
                doc = nlp(io.open(file_path, 'r').read())
                for sentence in doc.sents:
                    author_pred, same_prob, diff_prob = classifier.pred_text(sentence.text)
                    if same_prob > diff_prob and same_prob > args.threshold:
                        pass
                    else:
                        success += 1.0
                    # end if
                    count += 1.0
                    # end for
                # end for
            # end if
        # end for

        # >> 11. Save results
        logging.save_results("Success rate ", (success / count) * 100.0, display=True)
        logging.save_results("Max prob ", max_prob * 100, display=True)
        logging.save_results("Max cat ", max_cat, display=True)
        success_rates = np.append(success_rates, [(success / count) * 100.0])

        # Delete variables
        del classifier
    # end for

    # >> 10. Log success
    logging.save_results("Overall success rate ", np.average(success_rates), display=True)
    logging.save_results("Overall success rate std ", np.std(success_rates), display=True)

    # Plot histogram
    print("Plotting histogram of probabilities")
    bins = np.linspace(0, 1.0, 150)
    plt.hist(same_probs, bins, alpha=0.5, label="Same distrib")
    plt.hist(diff_probs, bins, alpha=0.5, label="Different distrib")
    plt.legend(loc='upper right')
    plt.show()

    # Plot histogram
    print("Plotting histogram of success rates")
    bins = np.linspace(0, 100, 150)
    plt.hist(success_rates, bins, label="Success rates")
    plt.legend(loc='upper right')
    plt.show()

# end if