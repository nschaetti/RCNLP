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
from scipy.stats import ttest_1samp
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter
import io
from sklearn.decomposition import PCA
import pickle

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Create PCA model of symbolic representations.")

    # Argument
    parser.add_argument("--texts", type=str, help="Text directory.")
    parser.add_argument("--startup", type=int, help="Number of start-up states to remove.", default=20)
    parser.add_argument("--components", type=int, help="Number of principal component to reduce inputs.", required=True)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).")
    parser.add_argument("--lang", type=str, help="Language model", default='en')
    parser.add_argument("--samples", type=int, help="Number of authors to take", default=20)
    parser.add_argument("--output", type=str, help="Output model file", default='pca_output.p')
    args = parser.parse_args()

    # >> 1. Convert the text to symbolic or continuous representations
    if args.converter == "pos":
        converter = RCNLPPosConverter()
    elif args.converter == "tag":
        converter = RCNLPTagConverter()
    elif args.converter == "fw":
        converter = RCNLPFuncWordConverter()
    else:
        converter = RCNLPWordVectorConverter()
    # end if

    # Get texts
    for i in np.arange(0, args.samples):
        # Choose authors and text
        authors_id = np.random.randint(1, 50)
        texts = os.path.join(args.texts, str(authors_id))

        # Generate states for first author
        print("Transforming texts from author %s to symbols" % texts)
        for index, text_file in enumerate(os.listdir(texts)):
            # Convert the text to Temporal Vector Representation
            doc_array = converter(io.open(os.path.join(texts, text_file), 'r').read())[args.startup:]

            # Add
            if i == 0:
                symb_rep = doc_array
            else:
                symb_rep = np.vstack((symb_rep, doc_array))
            # end if
        # end for
    # end for

    # PCA
    pca = PCA(n_components=args.components)
    pca.fit(symb_rep)

    # Explained variance
    print("Explained variance : ")
    print(pca.explained_variance_)

    # Explained variance ratio
    print("Explained variance ratio : ")
    print(pca.explained_variance_ratio_)

    # Mean
    print("Mean : ")
    print(pca.mean_)

    # Noise variance
    print("Noise variance : ")
    print(pca.noise_variance_)

    # Save
    pickle.dump(pca, open(args.output, 'w'))

# end if