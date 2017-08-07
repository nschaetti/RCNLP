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

import numpy as np
import spacy
from Converter import Converter
from core.embeddings.Word2Vec import Word2Vec
import scipy.sparse as sp


# Convert text to one-hot vectors
class OneHotConverter(Converter):
    """
    Convert text to one-hot vectors
    """

    # Constructor
    def __init__(self, lang='en', voc_size=5000):
        """
        Constructor
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        super(OneHotConverter, self).__init__(lang, None, -1, None)
        self._voc_size = voc_size
        self._word2vec = Word2Vec(dim=self._voc_size, mapper='one-hot')
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get word2vec
    def get_word2vec(self):
        """
        Get word2vec
        :return:
        """
        return self._word2vec
    # end get_word2vec

    ##############################################
    # Override
    ##############################################

    # Convert a string to a ESN input
    def __call__(self, text, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param text: The text to convert.
        :return: An numpy array of inputs.
        """
        # Load language model
        nlp = spacy.load(self._lang)

        # Process text
        if self._upper_level is not None:
            doc = nlp(self._upper_level(text))
        else:
            doc = nlp(text)
        # end if

        # Resulting numpy array
        doc_array = np.array([])

        # For each token
        ok = False
        for index, word in enumerate(doc):
            if word not in exclude:
                word_text = word.text
                word_text = word_text.replace(u"\n", u"")
                word_text = word_text.replace(u"\t", u"")
                word_text = word_text.replace(u"\r", u"")
                if len(word_text) > 0:
                    if not ok:
                        doc_array = self._word2vec[word_text]
                        ok = True
                    else:
                        doc_array = sp.vstack((doc_array, self._word2vec[word_text]))
                    # end if
                # end if
            # end if
        # end for

        # Del spacy nlp
        del nlp

        return self.reduce(doc_array)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Get the number of inputs
    def _get_inputs_size(self):
        """
        Get the input size.
        :return: The input size.
        """
        return self._voc_size
    # end get_n_inputs

    ##############################################
    # Static
    ##############################################

    # Generate data set inputs
    @staticmethod
    def generate_data_set_inputs(reps, n_authors, author):
        """
        Generate data set inputs
        :param reps:
        :param n_authors:
        :param author:
        :return:
        """
        # Number of representations
        n_reps = reps.shape[0]

        # Author vector
        outputs = sp.csr_matrix((n_reps, n_authors))
        outputs[:, author] = 1.0

        return reps, outputs
    # end generate_data_set_inputs

# end RCNLPConverter
