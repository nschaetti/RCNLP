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
import pickle
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter

#########################################################################
#
# Experience settings
#
#########################################################################

# Exp. info
ex_name = "Author clustering Experience"
ex_instance = "Author clustering One Reservoir"

# Reservoir Properties
rc_leak_rate = 0.5  # Leak rate
rc_input_scaling = 0.5  # Input scaling
rc_size = 500  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.2

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Author clustering with Part-Of-Speech to Echo State Network")

    # Argument
    parser.add_argument("--texts", type=str, help="Text directory.")
    parser.add_argument("--startup", type=int, help="Number of start-up states to remove.", default=20)
    parser.add_argument("--out-components", type=int, help="Number of principal component to reduce reservoir states.", default=-1)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs.", default=-1)
    parser.add_argument("--homogene", action='store_true', help="Keep the same number of states for each authors.",
                        default=False)
    parser.add_argument("--pca-images", action='store_true', help="Generate image of principal components.",
                        default=False)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).")
    parser.add_argument("--lang", type=str, help="Language model", default='en')
    parser.add_argument("--show-states", type=int, help="Number of states to show", default=500)
    parser.add_argument("--samples", type=int, help="Samples to estimate performances", default=20)
    parser.add_argument("--nreservoir", type=int, help="Number of reservoir to generate.", default=20)
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default='')
    parser.add_argument("--output", type=str, help="Where to save the reservoir.", default="reservoir.p")
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

    # Results
    state_results = np.array([])
    doc_results = np.array([])

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

    # Whathever
    max_score = 0.0

    # Iterate over reservoirs
    for r in range(args.nreservoir):
        # Create a reservoir
        flow = cf.create_reservoir(converter.get_n_inputs(), rc_size, rc_input_scaling, rc_leak_rate, rc_spectral_radius,
                                   rc_input_sparsity, rc_w_sparsity)

        # Iterate
        for i in np.arange(0, args.samples):
            authors_id = np.random.choice(49, 2, replace=False) + 1
            texts1 = os.path.join(args.texts, str(authors_id[0]))
            texts2 = os.path.join(args.texts, str(authors_id[1]))
            print("Round %d with author %s and %s" % (i, texts1, texts2))
            state_score, doc_score = cf.clustering_states(args=args, texts1=texts1, texts2=texts2, ex_name=ex_name,
                                                          ex_instance=ex_instance, input_scaling=rc_input_scaling,
                                                          input_sparsity=rc_input_sparsity, leak_rate=rc_leak_rate,
                                                          logging=logging, size=rc_size, spectral_radius=rc_spectral_radius,
                                                          w_sparsity=rc_w_sparsity, save_graph=True if i == 0 else False,
                                                          pca_model=pca_model, flow=flow)

            # Log results
            #logging.save_results("V state score " + str(i), state_score, display=True)
            #logging.save_results("V doc score " + str(i), doc_score, display=True)

            # Save results
            state_results = np.append(state_results, state_score)
            doc_results = np.append(doc_results, doc_score)
        # end for

        # Log overall state results
        print("#############################################################################")
        logging.save_results("Overall state score", np.average(state_results), display=True)
        logging.save_results("Overall state score std", np.std(state_results), display=True)
        logging.save_results("State score T-test", ttest_1samp(state_results, 0).pvalue * 100.0, display=True)

        # Log overall doc results
        logging.save_results("Overall doc score", np.average(doc_results), display=True)
        logging.save_results("Overall doc score std", np.std(doc_results), display=True)
        logging.save_results("Doc score T-test", ttest_1samp(doc_results, 0).pvalue * 100.0, display=True)

        # Save reservoir
        if np.average(doc_results) > max_score:
            pickle.dump(flow, open(args.output, 'w'))
        # end if
    # end for

    # Open logging dir
    logging.open_dir()

# end if