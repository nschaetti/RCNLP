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
    parser.add_argument("--author1", type=str, help="Author 1' ID.")
    parser.add_argument("--author2", type=str, help="Author 2's ID.")
    parser.add_argument("--samples", type=int, help="Number of samples to use to assess accuracy.", default=20)
    parser.add_argument("--training_size", type=int, help="Number of texts to train the model.", default=2)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
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
    average_doc_success_rate = np.array([])
    average_sen_success_rate = np.array([])

    # For each samples
    for s in range(0, args.samples):
        # >> 5. Prepare training and test set.
        training_set_indexes = np.arange(0, args.training_size, 1)
        test_set_indexes = np.arange(args.training_size, 100, 1)

        # >> 6. Create Echo Word Classifier
        classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                             input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                             spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

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
        doc_success = sen_success = 0.0
        doc_count = sen_count = 0.0
        for author_index, author_id in enumerate((args.author1, args.author2)):
            author_path = os.path.join(args.dataset, "total", author_id)
            for file_index in test_set_indexes:
                file_path = os.path.join(author_path, str(file_index) + ".txt")

                # Document success rate
                author_pred, _, _ = classifier.pred(file_path)
                if author_pred == author_index:
                    doc_success += 1.0
                # end if
                doc_count += 1.0

                # Sentence success rate
                nlp = spacy.load(args.lang)
                doc = nlp(io.open(file_path, 'r').read())
                for sentence in doc.sents:
                    sentence_pred, _, _ = classifier.pred_text(sentence.text)
                    if sentence_pred == author_index:
                        sen_success += 1.0
                    # end if
                    sen_count += 1.0
                # end for
            # end for
        # end for

        # >> 10. Log success
        logging.save_results("Document success rate ", (doc_success / doc_count) * 100.0, display=True)
        logging.save_results("Sentence success rate ", (sen_success / sen_count) * 100.0, display=True)

        # >> 11. Save results
        average_doc_success_rate = np.append(average_doc_success_rate, [(doc_success / doc_count) * 100.0])
        average_sen_success_rate = np.append(average_sen_success_rate, [(sen_success / sen_count) * 100.0])

        # Delete variables
        del classifier
    # end for

    # Log results
    logging.save_results("Doc. average success rate ", np.average(average_doc_success_rate), display=True)
    logging.save_results("Doc. success rate std ", np.std(average_doc_success_rate), display=True)
    logging.save_results("Sen. average success rate ", np.average(average_sen_success_rate), display=True)
    logging.save_results("Sen. success rate std ", np.std(average_sen_success_rate), display=True)

    # Plot histogram
    bins = np.linspace(0, 1.0, 100)
    plt.hist(average_doc_success_rate, bins, label="Doc. success rate distribution")
    plt.legend(loc='upper right')
    plt.show()

    # Plot histogram
    bins = np.linspace(0, 1.0, 100)
    plt.hist(average_sen_success_rate, bins, label="Sen. success rate distribution")
    plt.legend(loc='upper right')
    plt.show()

# end if