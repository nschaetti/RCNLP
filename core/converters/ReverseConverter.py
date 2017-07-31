#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.ReverseConverter.py
# Description : Inverse the tokens of a text.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 30.07.2017 16:41:00
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


# Inverse the token of a text
class ReverseConverter(Converter):
    """
    Inverse the tokens of a text
    """

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Override
    ##############################################

    # Inverse the tokens of a text
    def __call__(self, text, exclude=list(), word_exclude=list()):
        """
        Inverse the tokens of a text
        :param text: The text to convert.
        :return: An inverted text.
        """
        # Load language model
        nlp = spacy.load(self._lang)

        # Process text
        doc = nlp(text)

        # Tokens
        tokens = list()

        # For each token
        for word in doc:
            tokens.append(word.text)
        # end for
        tokens.reverse()

        # Del spacy nlp
        del nlp
        print(tokens)
        return u" ".join(tokens)
    # end convert

# end ReverseConverter
