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
import numpy as np
import spacy
import nltk
from core.converters.OneHotConverter import OneHotConverter
from core.classifiers.EchoWordClassifier import EchoWordClassifier
from core.tools.Metrics import Metrics
import logging
import json

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Identification"
ex_instance = "Elena Ferrante"

# Reservoir Properties
rc_leak_rate = 0.01  # Leak rate
rc_input_scaling = 1.0  # Input scaling
rc_size = 500  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
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
        description="RCNLP - Author identification experiment on IQLA-GIAT books")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory")
    parser.add_argument("--author", type=str, help="Author to identify", default="Ferrante")
    parser.add_argument("--lang", type=str, help="Language (en_core_web_md, ar, en, es, pt)", default='en_core_web_md')
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

    # Load authors data
    f = open(os.path.join(args.dataset, "authors.json"))
    authors_data = json.load(open(os.path.join(args.dataset, "authors.json")))

    # Check author exists
    if args.author not in authors_data.keys():
        logger.fatal(u"Author {} not found".format(args.author))
    # end if

    # Get author data
    author_books = authors_data[args.author]
    n_books = len(author_books)

    # Other authors
    other_authors = authors_data.keys().remove(args.author)

    # Create Echo Word Classifier
    classifier = EchoWordClassifier(classes=[0, 1], size=rc_size, input_scaling=rc_input_scaling,
                                    leak_rate=rc_leak_rate,
                                    input_sparsity=rc_input_sparsity, converter=converter,
                                    spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

    # Success rates
    success_rates = np.zeros(n_books)

    # k-Fold cross validation
    for k in range(0, n_books):
        # Prepare training and test set.
        test_set_indexes = list(author_books[k])
        training_set_indexes = author_books
        training_set_indexes.pop(k)

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
        print(test_set[0][1])
        pred, _ = classifier(test_set[0][0])
        classifier.debug()

        print(test_set[-1][1])
        pred, _ = classifier(test_set[-1][0])
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