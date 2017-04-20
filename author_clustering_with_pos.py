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

#########################################################################
#
# Experience settings
#
#########################################################################

# Exp. info
ex_name = "Author clustering Experience"
ex_instance = "Author clustering, PCA"

# Reservoir Properties
rc_leak_rate = 0.2  # Leak rate
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
    parser.add_argument("--author1", type=str, help="First author text directory.")
    parser.add_argument("--author2", type=str, help="Second author text directory.")
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
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # Main
    average_precision = cf.main(args=args, ex_name=ex_name, ex_instance=ex_instance, input_scaling=rc_input_scaling,
                                input_sparsity=rc_input_sparsity, leak_rate=rc_leak_rate, logging=logging, size=rc_size,
                                spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

    # Log
    logging.save_results("Average precision", average_precision, display=True)

    # Open logging dir
    logging.open_dir()

# end if