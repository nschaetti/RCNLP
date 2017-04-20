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
import Oger
import numpy as np
from sklearn.decomposition import PCA
from core.converters.RCNLPPosConverter import RCNLPPosConverter
from core.converters.RCNLPTagConverter import RCNLPTagConverter
from core.converters.RCNLPWordVectorConverter import RCNLPWordVectorConverter
from core.converters.RCNLPFuncWordConverter import RCNLPFuncWordConverter
from core.tools.RCNLPLogging import RCNLPLogging
from core.nodes.RCNLPWordReservoirNode import RCNLPWordReservoirNode
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator
import mdp
import matplotlib.pyplot as plt
import time
from scipy.cluster.vq import kmeans, vq
from skimage.draw import circle

#########################################################################
#
# Experience settings
#
#########################################################################

# Exp. info
ex_name = "PCA inputs reduction"
ex_instance = "PCA"

####################################################
# Function
####################################################

def pca_reduction(doc, title, ncomponents):
    # PCA
    pca = PCA(n_components=ncomponents)
    pca.fit(doc)

    # Generate PCA
    reduced = pca.transform(doc)

    # Show
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=2)
    plot.add_sub_plot(title=ex_instance + ", " + title + " inputs", x_label="Time", y_label="Inputs")
    plot.imshow(np.transpose(doc), cmap='Greys')
    plot.add_sub_plot(title=ex_instance + ", " + title + " reduced inputs", x_label="Time", y_label="Inputs")
    plot.imshow(np.transpose(reduced), cmap='Greys')
    logging.save_plot(plot)
# end pca_reduction

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Author clustering with Part-Of-Speech to Echo State Network")

    # Argument
    parser.add_argument("--text", type=str, help="Text file")
    parser.add_argument("--poscomponents", type=int, help="Number of principal component for POS")
    parser.add_argument("--tagcomponents", type=int, help="Number of principal component for tags")
    parser.add_argument("--wvcomponents", type=int, help="Number of principal component for word vectors")
    parser.add_argument("--fwcomponents", type=int, help="Number of principal component for function words")
    parser.add_argument("--nfile", type=int, help="Number of text files to analyze", default=-1)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # Reduce POS
    pca_reduction(RCNLPPosConverter()(io.open(args.text, 'r').read()), title="POS", ncomponents=args.poscomponents)

    # Reduce Tags
    pca_reduction(RCNLPTagConverter()(io.open(args.text, 'r').read()), title="Tags", ncomponents=args.tagcomponents)

    # Reduce Word vectors
    pca_reduction(RCNLPWordVectorConverter()(io.open(args.text, 'r').read()), title="Word vectors", ncomponents=args.wvcomponents)

    # Reduce FW
    pca_reduction(RCNLPFuncWordConverter()(io.open(args.text, 'r').read()), title="Function words", ncomponents=args.fwcomponents)

    # Open logging dir
    logging.open_dir()

# end if