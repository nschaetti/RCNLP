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
from RCNLPConverter import RCNLPConverter


class RCNLPPosConverter(RCNLPConverter):

    # Constructor
    def __init__(self, lang='en', pos_to_sym=[]):
        """
        Constructor
        :param lang:
        :param pos_to_sym:
        """
        # Base class
        super(RCNLPPosConverter, self).__init__(lang=lang)

        # Generate tag symbols
        if len(pos_to_sym) == 0:
            self._pos_symbols = RCNLPPosConverter.generate_pos_symbols()
        else:
            self._pos_symbols = pos_to_sym
        # end if
    # end __init__

    # Generate tag symbols
    @staticmethod
    def generate_pos_symbols():
        """
        Generate tag symbols
        :return:
        """
        result = dict()
        pos = [u"ADJ", u"ADP", u"ADV", u"CCONJ", u"DET", u"NOUN", u"NUM", u"PART", u"PRON", u"PROPN", u"PUNCT",
               u"SYM", u"VERB", u"X"]
        n_pos = len(pos)
        for index, p in enumerate(pos):
            result[p] = np.zeros(n_pos)
            result[p][index] = 1.0
        # end for
        return result
    # end _generate_tag_symbols

    # Get symbol from tag
    def pos_to_symbol(self, pos):
        """
        Get symbol from tag
        :param pos:
        :return:
        """
        if pos in self._pos_symbols.keys():
            return self._pos_symbols[pos]
        return None
    # end pos_to_symbol

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

        # For each words
        for index, word in enumerate(doc):
            if word.pos_ not in exclude and word not in word_exclude:
                sym = self.pos_to_symbol(word.pos_)
                if sym is not None:
                    if index == 0:
                        doc_array = sym
                    else:
                        doc_array = np.vstack((doc_array, sym))
                    # end if
                # end if
            # end if
        # end for

        return doc_array
    # end convert

# end RCNLPConverter