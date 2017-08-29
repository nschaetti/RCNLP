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
import pickle
from core.converters.FuncWordConverter import FuncWordConverter
from core.converters.WVConverter import WVConverter
from core.converters.PosConverter import PosConverter
from core.converters.TagConverter import TagConverter
from core.converters.OneHotConverter import OneHotConverter
from core.classifiers.EchoWordClassifier import EchoWordClassifier
import logging
from core.embeddings.Word2Vec import Word2Vec
from sklearn.manifold import TSNE
import pylab as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution"
ex_instance = "Two Authors One-hot representations"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = 0.5  # Input scaling
rc_size = 600  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
rc_w_sparsity = 0.05
rc_input_sparsity = 0.05

####################################################
# Functions
####################################################


def get_similar_documents(document_index, document_embeddings, distance_measure='cosine'):
    """

    :param document_index:
    :param document_embeddings:
    :param distance_measure:
    :return:
    """
    reverse = {'euclidian': False, 'cosine': True, 'cosine_abs': True}
    similarities = list()
    for n in range(document_embeddings.shape[1]):
        if n != document_index:
            document_embedding1 = document_embeddings[:, document_index].reshape(1, -1)
            document_embedding2 = document_embeddings[:, n].reshape(1, -1)
            if distance_measure == "euclidian":
                distance = euclidean(document_embedding1, document_embedding2)
            elif distance_measure == 'cosine_abs':
                distance = np.abs(cosine_similarity(document_embedding1, document_embedding2))
            else:
                distance = cosine_similarity(document_embedding1, document_embedding2)
            # end if
            # end if
            similarities.append((n, distance))
        # end if
    # end for

    # Sort
    similarities.sort(key=lambda tup: tup[1], reverse=reverse[distance_measure])

    return similarities
# end get_similar_documents

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="RCNLP - Compare the Echo Text Classifier to other models with two authors")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory")
    parser.add_argument("--output", type=str, help="Output image", required=True)
    parser.add_argument("--n-authors", type=int, help="Number of authors", default=10)
    parser.add_argument("--n-documents", type=int, help="Number of documents per authors", default=10)
    parser.add_argument("--lang", type=str, help="Language (en_core_web_md, ar, en, es, pt)", default='en_core_web_md')
    parser.add_argument("--verbose", action='store_true', help="Verbose mode", default=False)
    parser.add_argument("--voc-size", type=int, help="Vocabulary size", default=5000)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    parser.add_argument("--sparse", action='store_true', help="Sparse matrix?", default=False)
    parser.add_argument("--fig-size", type=float, help="Figure size (pixels)", default=1024.0)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv)", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to",
                        default=-1)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="RCNLP")

    # PCA model
    pca_model = None
    if args.pca_model is not None:
        pca_model = pickle.load(open(args.pca_model, 'r'))
    # end if

    # Choose a text to symbol converter.
    if args.converter == "pos":
        converter = PosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = TagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = FuncWordConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "wv":
        converter = WVConverter(resize=args.in_components, pca_model=pca_model)
    else:
        word2vec = Word2Vec(dim=args.voc_size, mapper='one-hot')
        converter = OneHotConverter(lang=args.lang, voc_size=args.voc_size, word2vec=word2vec)
    # end if

    # Total number of docs
    n_total_docs = args.n_authors * args.n_documents

    # Create Echo Word Classifier
    classifier = EchoWordClassifier(classes=range(n_total_docs), size=rc_size, input_scaling=rc_input_scaling,
                                    leak_rate=rc_leak_rate,
                                    input_sparsity=rc_input_sparsity, converter=converter,
                                    spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity,
                                    use_sparse_matrix=args.sparse)

    # Add examples
    document_index = 0
    for author_id in np.arange(1, args.n_authors+1):
        author_path = os.path.join(args.dataset, "total", str(author_id))
        for file_index in range(args.n_documents):
            file_path = os.path.join(author_path, str(file_index) + ".txt")
            logger.info(u"Adding document {} as {}".format(file_path, document_index))
            classifier.train(io.open(file_path, 'r').read(), document_index)
            document_index += 1
        # end for
    # end for

    # Finalize model training
    classifier.finalize(verbose=args.verbose)

    # Get documents embeddings
    document_embeddings = classifier.get_embeddings()
    logger.info(u"Document embeddings shape : {}".format(document_embeddings.shape))

    # Display similar doc for the first document of each author with each distance measure
    for distance_measure in ["euclidian", "cosine", "cosine_abs"]:
        print(u"###################### {} ######################".format(distance_measure))
        for document_index in np.arange(0, n_total_docs, args.n_authors):
            similar_doc = get_similar_documents(document_index, document_embeddings, distance_measure=distance_measure)
            logger.info(u"Documents similar to {} : {}".format(document_index, similar_doc[:10]))
        # end for
    # end for

    # Reduce with t-SNE
    model = TSNE(n_components=2, random_state=0)
    reduced_matrix = model.fit_transform(document_embeddings.T)

    # Word embedding matrix's size
    logger.info(u"Reduced matrix's size : {}".format(reduced_matrix.shape))

    # Show t-SNE
    plt.figure(figsize=(args.fig_size*0.003, args.fig_size*0.003), dpi=300)
    max_x = np.amax(reduced_matrix, axis=0)[0]
    max_y = np.amax(reduced_matrix, axis=0)[1]
    min_x = np.amin(reduced_matrix, axis=0)[0]
    min_y = np.amin(reduced_matrix, axis=0)[1]
    plt.xlim((min_x * 1.2, max_x * 1.2))
    plt.ylim((min_y * 1.2, max_y * 1.2))
    for document_index in range(n_total_docs):
        author_index = int(float(document_index) / float(args.n_authors))
        plt.scatter(reduced_matrix[document_index, 0], reduced_matrix[document_index, 1], 0.5)
        plt.text(reduced_matrix[document_index, 0], reduced_matrix[document_index, 1], str(document_index) + ", " + str(author_index), fontsize=2.5)
    # end for

    # Save image
    plt.savefig(args.output)

# end if
