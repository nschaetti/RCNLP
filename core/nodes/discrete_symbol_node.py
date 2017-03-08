#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPDiscreteSymbolNode.py
# Description : Discrete symbol node class.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
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
try:
    import cudamat as cm
except:
    pass


class RCNLPDiscreteSymbolNode(mdp.Node):
    """
    This class put the maximum output to one and the
    other to zero.
    """

    # Constructor
    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        """
        Constructor
        :param input_dim:
        :param output_dim:
        :param dtype:
        """
        super(DiscreteSymbolNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        self._is_initialized = False

        if input_dim is not None and output_dim is not None:
            # Call the initialize function to create the weight matrices
            self.initialize()


    # Override the standard output_dim getter and setter property, 
    # to enable changing the output_dim (i.e. the number
    # of neurons) afterwards during optimization
    def get_output_dim(self): 
        return self._output_dim

    def set_output_dim(self, value): 
        self._output_dim = value
    output_dim = property(get_output_dim, set_output_dim, doc="Output dimensions")

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def initialize(self):

        # Check input dim
        if self.input_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: input_dim is not set.')

        # Check output dim
        if self.output_dim is None:
            raise mdp.NodeException('Cannot initialize weight matrices: output_dim is not set.')

        self._is_initialized = True


    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, s):
        """ Put max symbol output to one, other to zero.
        """

        # Check if the weight matrices are intialized, otherwise create them
        if not self._is_initialized:
            self.initialize()

        # Foreach inputs sets
        for symbol_set in s:
            symbol_set[symbol_set != np.max(symbol_set)] = 0
            symbol_set[symbol_set == np.max(symbol_set)] = 1
        
        # Return the whole state matrix except the initial state
        return s

    def _post_update_hook(self, states, input, timestep):
        """ Hook which gets executed after the state update equation for every timestep. Do not use this to change the state of the 
            reservoir (e.g. to train internal weights) if you want to use parallellization - use the TrainableReservoirNode in that case.
        """
        pass
