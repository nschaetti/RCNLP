#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPContainerConverter.py
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
from Converter import Converter


class JoinConverter(Converter):
    """
    Join two converters
    """

    # Constructor
    def __init__(self, conv1, conv2, lang='en', tag_to_symbol=None, resize=-1, pca_model=None):
        """
        Constructor
        :param conv1: First converter
        :param conv2: Second converter
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        super(JoinConverter, self).__init__(lang, tag_to_symbol, resize, pca_model)

        # Converters
        self._conv1 = conv1
        self._conv2 = conv2
    # end __init__

    # Get the number of inputs
    def _get_inputs_size(self):
        """
        Get the input size.
        :return: The input size.
        """
        return self._conv1.get_n_inputs() + self._conv2.get_n_inputs()
    # end get_n_inputs

    # Convert a string to a ESN input
    def __call__(self, text, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param text: The text to convert.
        :return: An numpy array of inputs.
        """

        # Converted text 1
        conv1_text = self._conv1(text, exclude, word_exclude)

        # Converted text 2
        conv2_text = self._conv2(text, exclude, word_exclude)

        return np.hstack((conv1_text, conv2_text))
    # end convert

# end RCNLPConverter
