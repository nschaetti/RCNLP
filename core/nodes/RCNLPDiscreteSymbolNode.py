#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPDiscreteSymbolNode.py
# Description : Discrete symbol node class.
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

# Package import
import mdp.utils
import numpy as np


class RCNLPDiscreteSymbolNode(mdp.Node):
    """
    This class put the maximum output to one and the
    other to zero.
    """

    # Constructor
    def __init__(self, nb_symbols, dtype='float64'):
        """
        Constructor
        :param input_dim: The number of
        :param output_dim:
        :param dtype:
        """
        super(RCNLPDiscreteSymbolNode, self).__init__(input_dim=nb_symbols, output_dim=nb_symbols, dtype=dtype)

        # Not initialized
        self._is_initialized = False

        # Check input and output dims
        if input_dim is not None and output_dim is not None:
            # Call the initialize function to create the weight matrices
            self.initialize()
        # endif
    # end __init__

    # Return the output dimension
    def get_output_dim(self):
        """
        Return the output dimension
        :return: The output dimension.
        """
        return self._output_dim
    # end get_output_dim

    # Change the output dimension
    def set_output_dim(self, value):
        """
        Change the output dimension
        :param value:
        :return:
        """
        self._output_dim = value
        output_dim = property(get_output_dim, set_output_dim, doc="Output dimensions")
    # end set_output_dim

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
            symbol_set[symbol_set != np.max(symbol_set)] = 0
            symbol_set[symbol_set == np.max(symbol_set)] = 1
        # endfor
        
        # Return the whole state matrix except the initial state
        return s
    # end _execute

# end RCNLPDiscreteSymbolNode
