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
import pickle
import numpy as np
import spacy
from scipy import stats
from core.converters.PosConverter import PosConverter
from core.converters.TagConverter import TagConverter
from core.converters.WVConverter import WVConverter
from core.converters.FuncWordConverter import FuncWordConverter
from core.converters.OneHotConverter import OneHotConverter
from core.classifiers.EchoWordClassifier import EchoWordClassifier
from core.tools.RCNLPLogging import RCNLPLogging
from core.tools.Metrics import Metrics
import logging

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution"
ex_instance = "Two Authors One-hot representations"

# Reservoir Properties
rc_leak_rate = 0.5  # Leak rate
rc_input_scaling = 0.5  # Input scaling
rc_size = 500  # Reservoir size
rc_spectral_radius = 0.1  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.01
sl_smoothing_param = 0.5

####################################################
# Functions
####################################################

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="RCNLP - Compare the Echo Text Classifier to other models with two authors")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory")
    parser.add_argument("--author1", type=str, help="First author", default="1")
    parser.add_argument("--author2", type=str, help="Second author", default="2")
    parser.add_argument("--lang", type=str, help="Language (en_core_web_md, ar, en, es, pt)", default='en_core_web_md')
    parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?", default=False)
    parser.add_argument("--k", type=int, help="n-Fold Cross Validation", default=10)
    parser.add_argument("--verbose", action='store_true', help="Verbose mode", default=False)
    parser.add_argument("--debug", action='store_true', help="Debug mode", default=False)
    parser.add_argument("--voc-size", type=int, help="Vocabulary size", default=5000, required=True)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="RCNLP")

    # Choose a text to symbol converter
    converter = OneHotConverter(lang=args.lang, voc_size=args.voc_size)

    # Prepare training and test set indexes.
    n_fold_samples = int(100 / args.k)
    indexes = np.arange(0, 100, 1)
    indexes.shape = (args.k, n_fold_samples)

    # Create Echo Word Classifier
    classifier = EchoWordClassifier(classes=[0, 1], size=rc_size, input_scaling=rc_input_scaling,
                                    leak_rate=rc_leak_rate,
                                    input_sparsity=rc_input_sparsity, converter=converter,
                                    spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

    # Success rates
    success_rates = np.zeros(args.k)

    # k-Fold cross validation
    for k in range(0, args.k):
        # Prepare training and test set.
        test_set_indexes = indexes[k]
        training_set_indexes = indexes
        training_set_indexes = np.delete(training_set_indexes, k, axis=0)
        training_set_indexes.shape = (100 - n_fold_samples)

        # Add examples
        for author_index, author_id in enumerate((args.author1, args.author2)):
            author_path = os.path.join(args.dataset, "total", author_id)
            for file_index in training_set_indexes:
                file_path = os.path.join(author_path, str(file_index) + ".txt")
                classifier.train(io.open(file_path, 'r').read(), author_index)
            # end for
        # end for

        # Finalize model training
        classifier.finalize(verbose=args.verbose)

        # Init test epoch
        test_set = list()

        # Get text
        for author_index, author_id in enumerate((args.author1, args.author2)):
            author_path = os.path.join(args.dataset, "total", str(author_id))
            for file_index in test_set_indexes:
                file_path = os.path.join(author_path, str(file_index) + ".txt")
                # Document success rate
                if not args.sentence:
                    test_set.append((io.open(file_path, 'r').read(), author_index))
                else:
                    # Sentence success rate
                    nlp = spacy.load(args.lang)
                    doc = nlp(io.open(file_path, 'r').read())
                    for sentence in doc.sents:
                        test_set.append((sentence, author_index))
                    # end for
                # end if
            # end for
        # end for

        # Classify
        pred, _ = classifier(test_set[0])
        classifier.debug()
        exit()

        # Success rate
        success_rate = Metrics.success_rate(classifier, test_set, verbose=args.verbose, debug=args.debug)
        logger.info(u"\t{} - Success rate : {}".format(k, success_rate))

        # Save result
        success_rates[k] = success_rate
    # end for

    # Over all success rate
        logger.info(u"All - Success rate : {}".format(np.average(success_rates)))

# end if