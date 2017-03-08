#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPRememberSymbolsLanguage.py
# Description : Remember symbols language class.
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
import RCNLPLanguage


#
# CLASS RememberSymbols
#
class RCNLPRememberSymbolsLanguage(RCNLPLanguage):

    # Constructeur
    def __init__(self, n_symbols):
        """

        :param n_symbols:
        """
        self.m_sTagSymbol = tag_symbol  # Symbol to remember
        self.m_aOtherSymbols = other_symbols  # Other symbols taken randomly
        self.m_iMemoryLength = memory_length  # Number of step we keep the tag symbol
        self.m_bSloppingMemory = slopping_memory  # Is the memory slopping?
        self.m_nb_symbols = n_symbols
    # end __init__

    # Generate a sample
    def generate(self, length=1000):
        """

        :param length:
        :param tag_places:
        :return:
        """
        # Input output
        inputs = []
        outputs = []

        # Memory step
        if self.m_bSloppingMemory:
            mem_step = 1.0 / self.m_iMemoryLength
        else:
            mem_step = 0.0

        # Generate each symbols
        count = 0
        seen = 0
        seen_pos = -1
        for i in range(length):

            if count in tag_places:
                inputs = inputs + [self.m_sTagSymbol]
                seen = 1.0
                seen_pos = count
            else:
                sym_pos = np.random.randint(0, len(self.m_aOtherSymbols))
                inputs = inputs + [self.m_aOtherSymbols[sym_pos]]

            outputs = outputs + [[seen]]

            if seen_pos != -1 and self.m_iMemoryLength != -1 and count - seen_pos >= self.m_iMemoryLength - 1:
                seen_pos = -1
                seen = 0.0
            elif seen_pos != -1 and self.m_iMemoryLength != -1:
                seen -= mem_step

            count += 1

        return (inputs, outputs)
    # end generate

    # Generate the whole data set
    def generate_data_set(self, n_samples=10, sample_length=1000):
        """

        :param n_samples:
        :param sample_length:
        :return:
        """
        # Inputs/ouputs
        inputs = []
        outputs = []

        # For each samples
        for n_sample in range(n_samples):

            # Sparse sample or not
            if np.random.random_sample() <= sparsity:
                inp, out = self.generate(sample_length, [sample_length + 10])
            else:
                inp, out = self.generate(sample_length, [np.random.randint(0, sample_length - self.m_iMemoryLength * 1.5)])

            # Add
            inputs = inputs + [inp]
            outputs = outputs + [out]

        return np.array(inputs), np.array(outputs)
    # end generate_data_set
