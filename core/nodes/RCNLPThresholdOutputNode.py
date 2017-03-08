#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPThresholdOutputNode.py
# Description : Threshold output node class.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 22.02.2017 17:59:05
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

# Package import
import mdp.utils
import numpy as np


class RCNLPThresholdOutputNode(mdp.Node):
    """
    This class put the output to one if they
    exceed a threshold
    """

    # Constructor
    def __init__(self, size, threshold=1.0, use_abs=False, dtype='float64'):
        """
        Constructor
        :param threshold: The output threshold
        :param use_abs: Do we put outputs through abs before?
        :param dtype: Node data type
        """
        super(RCNLPThresholdOutputNode, self).__init__(input_dim=size, output_dim=size, dtype=dtype)

        # Variables
        self.threshold = threshold
        self.use_abs = use_abs

        # Not initialized
        self._is_initialized = False

        # Check input and output dims
        if self.input_dim is not None and self.output_dim is not None:
            # Call the initialize function to create the weight matrices
            self.initialize()
            # endif

    # end __init__

    # Node initialization
    def initialize(self):
        """
        Node initialization
        :return: None
        """
        # Check input dim
        if self.input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        # Check output dim
        if self.output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')
        self._is_initialized = True

    # end initialize

    # Not trainable
    def is_trainable(self):
        """
        This node is not trainable.
        :return: False
        """
        return False

    # end is_trainable

    # Not inversible
    def is_invertible(self):
        """
        This node is not invertible.
        :return: False
        """
        return False

    # end is_invertible

    # Get supported data types
    def _get_supported_dtypes(self):
        """
        Get the supported data types.
        :return: An array of supported data types.
        """
        return ['float32', 'float64']

    # end _get_supported_dtypes

    # Compute one input.
    def _execute(self, s):
        """
        Compute one input.
        :param s: The input
        :return: The output with max element set to one other to zero.
        """

        # Check if the weight matrices are intialized, otherwise create them
        if not self._is_initialized:
            self.initialize()

        # Foreach inputs sets
        for symbol_set in s:
            if self.use_abs:
                symbol_set[abs(symbol_set) >= self.threshold] = 1
                symbol_set[abs(symbol_set) < self.threshold] = 0
            else:
                symbol_set[symbol_set >= self.threshold] = 1
                symbol_set[symbol_set < self.threshold] = 0
        # endfor

        # Return the whole state matrix except the initial state
        return s
    # end _execute

# end RCNLPDiscreteSymbolNode
