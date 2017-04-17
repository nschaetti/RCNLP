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
import scipy.signal as sig
from RCNLPConverter import RCNLPConverter


class RCNLPWordVectorConverter(RCNLPConverter):

    # Constructor
    def __init__(self, lang='en', resize=300):
        """
        Constructor
        :param lang:
        :param resize:
        """
        # Base class
        super(RCNLPWordVectorConverter, self).__init__(lang=lang)
        self._resize = resize
    # end __init__

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

        for index, word in enumerate(doc):
            if word not in exclude:
                word_vector = word.vector
                if self._resize != 300:
                    word_vector = sig.resample(word_vector, self._resize)
                # end if
                if index == 0:
                    doc_array = word_vector
                else:
                    doc_array = np.vstack((doc_array, word_vector))
                # end if
            # end if
        # end for
        return doc_array
    # end convert

# end RCNLPConverter