#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPMultidimensionalMemoryLanguage.py
# Description : Multidimensional memory language class.
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


#
# CLASS RCNLPMultidimensionalMemoryLanguage
#
class RCNLPMultidimensionalMemoryLanguage:

    # Constructeur
    def __init__(self, n_dim, memory_shifting):
        """

        :param n_dim:
        :param memory_shifting:
        """
        pass
    # end __init__
    
    # Génère une chaîne
    def generate(self, length=1000):
        """
        Generate a sample of the language.
        :param n_dim:
        :param memory_shiffting:
        :param length:
        :return:
        """
        # Input output
        inputs = np.array([])
        inputs.shape = (0, n_dim)
        outputs = np.zeros((memory_shiffting, n_dim))
        
        # For each symbols to remember
        for i in range(length - memory_shiffting):
            rand_inputs = np.random.rand(n_dim)
            inputs = np.vstack((inputs, rand_inputs))
            outputs = np.vstack((outputs, rand_inputs))

        # No symbol to remember at the end
        inputs = np.vstack((inputs,np.zeros((memory_shiffting, n_dim))))
        
        return (inputs, outputs)
    # end generate
    
    # Génère plusieurs chaine
    def generate_data_set(self, n_samples=10, sample_length=1000):
        """
        Generate the whole data set for the language.
        :param n_dim:
        :param memory_shiffting:
        :param n_samples:
        :param sample_length:
        :return:
        """
        # Inputs/ouputs
        inputs = []
        outputs = []
        
        # For each samples
        for n_sample in range(n_samples):
            
            # Generate
            inp, out = self.generate(n_dim, memory_shiffting, length = sample_length)
            
            # Add
            inputs = inputs + [inp]
            outputs = outputs + [out]
        
        return np.array(inputs), np.array(outputs)
    # end generate_data_set

# end RCNLPMultidimensionalMemoryLanguage
