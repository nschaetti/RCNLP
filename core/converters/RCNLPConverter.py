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
import matplotlib.pyplot as plt


class RCNLPConverter(object):

    # Constructor
    def __init__(self, lang='en'):
        """
        Constructor.
        :param lang:
        """
        self._lang = lang
    # end __init__

    # Convert a string to a ESN input
    def __call__(self, text, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param text: The text to convert.
        :return: An numpy array of inputs.
        """
        pass
    # end convert

    # Display representations
    @staticmethod
    def display_representations(rep):
        """
        Display representations
        :param rep:
        :return:
        """
        plt.imshow(rep, cmap='Greys')
        plt.show()
    # end display_representations

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
        author_vector = np.zeros((1, n_authors))
        author_vector[0, author] = 1.0

        # Output
        outputs = np.repeat(author_vector, n_reps, axis=0)

        return zip(reps, outputs)
    # end generate_data_set_inputs

# end RCNLPConverter