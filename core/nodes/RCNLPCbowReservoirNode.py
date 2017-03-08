#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPCbowReservoirNode.py
# Description : CBOW reservoir node class.
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

import Oger
import mdp.utils
import numpy as np


class RCNLPCbowReservoirNode(Oger.nodes.LeakyReservoirNode):
    """Reservoir node with leaky integrator neurons to handle distributed representation of words. 
    """

    # Constructor
    def __init__(self, nb_words = 1.0, word_sparsity=1.0, *args, **kwargs):

        super(RCNLPCbowReservoirNode, self).__init__(*args, **kwargs)
        
        # Number if neurons linked to each inputs
        if self.input_dim > self.output_dim:
            raise Exception("Words number is higher than the reservoir size")
        elif self.input_dim == self.output_dim:
            nb_neurons = 1
        else:
            nb_neurons = int(self.output_dim / self.input_dim * word_sparsity)
        
        # Permutations
        permuts = np.random.choice(self.output_dim, nb_neurons * self.input_dim, replace=False)
        permuts.shape = (self.input_dim, nb_neurons)
        
        # Init
        self.w_in = mdp.numx.zeros((self.output_dim, self.input_dim))
        
        # For each inputs
        n_neur = 0
        for inp in permuts:
            self.w_in[inp, n_neur] = mdp.numx.random.randint(0, 2, (nb_neurons)) * 2 - 1
            n_neur += 1
        # endfor

        self.w_in *= self.input_scaling
    # end __init__
