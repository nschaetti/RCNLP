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
import mdp
import matplotlib.pyplot as plt
import json
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


def load_truth(truth_file):
    return json.load(open(truth_file, 'r'))
# end load_truth


def to_filename(index):
    return "{04d}".format(index)
# end to_filename

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship clustering with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory")
    parser.add_argument("--problem", type=str, help="Problem's directory")
    parser.add_argument("--negatives", type=int, help="Number of negative texts to use", default=1)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--threshold", type=float, help="Confidence threshold", default=0.5)
    parser.add_argument("--gephi", type=str, help="Output Gephi file", default="output.gephi")
    parser.add_argument("--matrix", type=str, help="Output similarity matrix file", default="matrix.p")
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

    # >> 4. Generate W
    w = mdp.numx.random.choice([0.0, 1.0], (rc_size, rc_size), p=[1.0 - rc_w_sparsity, rc_w_sparsity])
    w[w == 1] = mdp.numx.random.rand(len(w[w == 1]))

    # Preparation
    data_set_files = os.listdir(os.path.join(args.dataset, args.problem))
    n_files = len(data_set_files)
    similarity_matrix = np.zeros((n_files, n_files))

    # For each file
    index1 = 0
    for text1 in os.listdir(os.path.join(args.dataset, args.problem)):
        text1_path = os.path.join(args.dataset, args.problem, text1)
        print(text1_path)
        index2 = 0
        for text2 in os.listdir(os.path.join(args.dataset, args.problem)):
            if text1 != text2:
                text2_path = os.path.join(args.dataset, args.problem, text2)

                # >> 6. Create Echo Word Classifier
                classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling,
                                                     leak_rate=rc_leak_rate,
                                                     input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                                     spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity, w=w)

                # >> 7. Add authors examples
                classifier.add_example(text1_path, 0)

                # >> 8. Add negative examples
                others_path = os.path.join(args.dataset, "total", "others")
                for file_index in range(0, args.negatives):
                    file_path = os.path.join(args.dataset, "others", str(file_index) + ".txt")
                    classifier.add_example(file_path, 1)
                # end for

                # >> 8. Train model
                classifier.train()

                # Get similarity
                author_pred, same_prob, diff_prob = classifier.pred(text2_path)
                print("%s : %f" % (text2, same_prob))
                # Save
                similarity_matrix[index1, index2] = same_prob
            # end if
            index2 += 1
        # end for
        index1 += 1
    # end for

    plt.imshow(similarity_matrix, cmap='gray')
    plt.show()
    pickle.dump(similarity_matrix, open(args.matrix, 'w'))

    # Get links
    count_links = 0
    links = dict()
    for index1 in np.arange(0, n_files, 1):
        links[to_filename(index1)] = list()
        for index2 in np.arange(0, n_files, 2):
            if index1 != index2:
                similarity = np.average([similarity_matrix[index1, index2], similarity_matrix[index2, index1]])
                if similarity > args.threshold:
                    print("Link found between %d and %d" % (index1, index2))
                    links[to_filename(index1)] = (to_filename(index2), similarity)
                    links[to_filename(index1)] = sorted(links[to_filename(index1)], key=lambda tup: tup[1])
                    count_links += 1
                # end if
            # end for
        # end for
    # end for

    # Load truth


    print(links)

# end if