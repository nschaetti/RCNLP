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


class RCNLPPosConverter(Converter):
    """
    Convert text to Part-Of-Speech symbols.
    """

    # Constructor
    def __init__(self, lang='en', tag_to_symbol=None, resize=-1, pca_model=None, fill_in=False):
        """
        Constructor
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        super(RCNLPPosConverter, self).__init__(lang, tag_to_symbol, resize, pca_model)
        self._fill_in = fill_in
    # end __init__

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A list of tags.
        """
        return [u"ADJ", u"ADP", u"ADV", u"CCONJ", u"DET", u"INTJ", u"NOUN", u"NUM", u"PART", u"PRON", u"PROPN", u"PUNCT",
               u"SYM", u"VERB", u"X"]
    # end get_tags

    # Get the number of inputs
    def _get_inputs_size(self):
        """
        Get the number of inputs.
        :return: The number of inputs.
        """
        return len(self.get_tags())
    # end get_n_inputs

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
        doc = nlp(text)

        # Resulting numpy array
        doc_array = np.array([])

        # Null symbol
        null_symbol = np.zeros((1, len(self.get_tags())))

        # For each words
        for index, word in enumerate(doc):
            if word.pos_ not in exclude and word not in word_exclude:
                sym = self.tag_to_symbol(word.pos_)
                if sym is None and self._fill_in:
                    sym = null_symbol
                # end if
                if sym is not None:
                    if index == 0:
                        doc_array = sym
                    else:
                        doc_array = np.vstack((doc_array, sym))
                    # end if
                # end if
            # end if
        # end for

        return self.reduce(doc_array)
    # end convert

# end RCNLPConverter