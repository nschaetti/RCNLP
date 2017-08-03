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
    def __init__(self, lang='en', tag_to_symbol=None, resize=-1, pca_model=None, fill_in=False):
        """
        Constructor
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        super(OneHotConverter, self).__init__(lang, tag_to_symbol, resize, pca_model)
        self._word2vec = None
    # end __init__

    ##############################################
    # Public
    ##############################################

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

        # Word2Vec
        self._word2vec = Word2Vec(dim=5000, mapper='one-hot')

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
        return self._word2vec.get_dimension()
    # end get_n_inputs

# end RCNLPConverter
