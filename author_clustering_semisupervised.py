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
from core.clustering.RCNLPEchoWordClustering import RCNLPEchoWordClustering
from core.tools.RCNLPLogging import RCNLPLogging
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Semi-supervised Authorship Clustering Experience"
ex_instance = "Author Clustering"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 10000  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.025

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Author clustering with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--n-authors", type=int, help="Number of authors to use for training.")
    parser.add_argument("--n-texts", type=int, help="Number of texts per authors to use for training.")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--model", type=str, help="ESN model to load", default='')
    parser.add_argument("--output", type=str, help="Output filename where to save the model.", default=None)
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    if args.model == '':
        # PCA model
        pca_model = None
        if args.pca_model is not None:
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

        # >> 3. Create Echo Word Classifier
        clustering = RCNLPEchoWordClustering(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                             input_sparsity=rc_input_sparsity, converter=converter,
                                             spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

        # >> 4. Add examples with same authors
        print("Adding examples with same authors...")
        for author_id in np.arange(1, args.n_authors+1, 1):
            # Author path
            author_path = os.path.join(args.dataset, "total", str(author_id))

            # For each texts
            for n in range(args.n_texts):
                # Random texts
                texts_id = np.random.randint(0, 99, 2)

                # Texts path
                text1_path = os.path.join(author_path, str(texts_id[0]) + ".txt")
                text2_path = os.path.join(author_path, str(texts_id[1]) + ".txt")

                # Add
                print("Adding examples %s and %s as same author texts." % (text1_path, text2_path))
                clustering.add_same_author_example(text1_path, text2_path)
            # end for
        # end for

        # >> 5. Add examples with different authors
        print("Adding examples with different authors...")
        for author1_id in np.arange(1, args.n_authors + 1, 1):
            # Other authors
            author2_id = np.random.choice(50, 1)[0] + 1

            # No same author
            if author1_id == author2_id:
                author2_id = np.random.choice(50, 1)[0] + 1
            # end if

            # Author path
            author1_path = os.path.join(args.dataset, "total", str(author1_id))
            author2_path = os.path.join(args.dataset, "total", str(author2_id))

            # For each texts
            for n in range(args.n_texts):
                # Random texts
                texts_id = np.random.randint(0, 99, 2)

                # Texts path
                text1_path = os.path.join(author1_path, str(texts_id[0]) + ".txt")
                text2_path = os.path.join(author2_path, str(texts_id[1]) + ".txt")

                # Add
                print("Adding examples %s and %s as different author texts." % (text1_path, text2_path))
                clustering.add_different_author_example(text1_path, text2_path)
            # end for
        # end for

        # >> 6. Train model
        print("Training model with text files from %s" % os.path.join(args.dataset, "total"))
        clustering.train()

        # >> 7. Save model
        pickle.dump(clustering, open(args.output, 'w'))
    else:
        print("Loading reservoir model %s" % args.model)
        clustering = pickle.load(open(args.model, 'r'))
    # end if

    # >> 8. Init counter
    success = 0.0
    count = 0.0
    same_probs = list()
    different_probs = list()

    # >> 9. Test model performance with same author
    show_graph = True
    for author_id in np.arange(args.n_authors+1, 51, 1):
        # Author path
        author_path = os.path.join(args.dataset, "total", str(author_id))
        print("For author %s" % author_path)

        # For each texts
        for n in range(args.n_texts):
            # Random texts
            texts_id = np.random.randint(0, 99, 2)

            # Texts path
            text1_path = os.path.join(author_path, str(texts_id[0]) + ".txt")
            text2_path = os.path.join(author_path, str(texts_id[1]) + ".txt")

            # Prediction
            same, prob, prob_std = clustering.pred(text1_path, text2_path, show_graph)
            print("[Same] Same : %r, Prob : %f, std : %f" % (same, prob, prob_std))
            show_graph = False

            # Add
            if same:
                success += 1.0
            # end if

            same_probs.append(prob)
            count += 1.0
        # end for
    # end for

    # >> 10. Test model performance with different authors.
    show_graph = True
    for author1_id in np.arange(args.n_authors + 1, 51, 1):
        # Other authors
        author2_id = np.random.choice(50, 1)[0] + 1

        # No same author
        if author1_id == author2_id:
            author2_id = np.random.choice(50, 1)[0] + 1
        # end if

        # Author path
        author1_path = os.path.join(args.dataset, "total", str(author1_id))
        author2_path = os.path.join(args.dataset, "total", str(author2_id))
        print("For authors %s and %s" % (author1_path, author2_path))

        # For each texts
        for n in range(args.n_texts):
            # Random texts
            texts_id = np.random.randint(0, 99, 2)

            # Texts path
            text1_path = os.path.join(author1_path, str(texts_id[0]) + ".txt")
            text2_path = os.path.join(author2_path, str(texts_id[1]) + ".txt")

            # Prediction
            same, prob, prob_std = clustering.pred(text1_path, text2_path, show_graph)
            print("[Diff] Same : %r, Prob : %f, std : %f" % (same, prob, prob_std))
            show_graph = False

            # Test
            if not same:
                success += 1.0
            # end for

            different_probs.append(prob)
            count += 1.0
        # end for
    # end for

    # Result
    logging.save_results("Same mu", np.average(same_probs), display=True)
    logging.save_results("Diff mu", np.average(different_probs), display=True)
    logging.save_results("TTest same mu = 0.5", ttest_1samp(same_probs, 0.5).pvalue * 100.0, display=True)
    logging.save_results("TTest diff mu = 0.5", ttest_1samp(different_probs, 0.5).pvalue * 100.0, display=True)
    logging.save_results("TTest diff mu = same mu", ttest_ind(different_probs, same_probs).pvalue * 100.0, display=True)
    logging.save_results("Success rate", success / count, display=True)

    # Plot histogram
    print("Plotting histogram")
    bins = np.linspace(0, 1.0, 100)
    plt.hist(same_probs, bins, alpha=0.5, label="Same distrib")
    plt.hist(different_probs, bins, alpha=0.5, label="Different distrib")
    plt.legend(loc='upper right')
    plt.show()
    print("end")

# end if