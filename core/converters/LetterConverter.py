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

# Imports
import numpy as np
from Converter import Converter


# Converter from letter to symbols
class LetterConverter(Converter):
    """
    Convert letter to symbols
    """

    # Constructor
    def __init__(self, lang='en', tag_to_symbol=None, resize=-1, pca_model=None, fill_in=False):
        """
        Constructor
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        super(LetterConverter, self).__init__(lang, tag_to_symbol, resize, pca_model)
        self._fill_in = fill_in
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A tag list.
        """
        return [u' ', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p',
                u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'.', u',', u';', u'-', u'!', u'?']
    # end get_tags

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
        # Resulting numpy array
        doc_array = np.array([])

        # Null symbol
        null_symbol = np.zeros((1, len(self.get_tags())))

        # For each letter
        init = False
        for index, letter in enumerate(text):
            sym = self.tag_to_symbol(letter)
            if sym is None and self._fill_in:
                sym = null_symbol
            # end if
            if sym is not None:
                if not init:
                    doc_array = sym
                    init = True
                else:
                    doc_array = np.vstack((doc_array, sym))
                    # end if
                    # end if
        # end for

        return self.reduce(doc_array)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return: The input size.
        """
        return len(self.get_tags())
    # end if

# end LetterConverter
