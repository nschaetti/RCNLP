#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : RCNLPSwitchingAttractorLanguage.py
# Description : Switching attractor language class.
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

from RCNLPLanguage import RCNLPLanguage
import numpy as np


#
# CLASS RCNLPSwitchingAttractorLanguage
#
class RCNLPSwitchingAttractorLanguage(RCNLPLanguage):
    """
    This class create samples of a language where the reservoir must
    switch attractor and therefore its outputs when a signal is
    presented to its input.
    """

    # Constructor
    def __init__(self, tag_symbol, other_symbols, switch_back_symbol=None, memory_length=-1, sparsity=0.5):
        """
        :param tag_symbol: The symbol tag to switch attractor
        :param other_symbols: The other symbol given to the reservoir
        :param memory_length: The number of step the reservoir must stay in the attractor.
        """
        super(RCNLPSwitchingAttractorLanguage, self).__init__()

        # Variables
        self.m_tag_symbol = tag_symbol						# Symbol to switch attractor
        self.m_other_symbols = other_symbols				# Other symbols taken randomly
        self.m_memory_length = memory_length				# Number of step the reservoir must stay in the attractor (-1 is infinite)
        self.m_sparsity = sparsity                          # Portion of sample with no switching
        self._switch_back_symbol = switch_back_symbol

    # end __init__

    # Generate a sample
    def generate(self, length=1000, tag_places=[0]):
        """
        Generate a sample.
        :param length: The sample's length.
        :param tag_places : The position where the attractor must switch.
        :return: A sample of the language.
        """

        # Input output
        lang_inputs = []
        lang_outputs = []

        # Memory step
        mem_step = 0.0

        # Generate each symbols
        count = 0
        seen = 0
        seen_pos = -1

        # For each element to produce
        for i in range(length):

            # If this is a tag pos or not
            if count in tag_places:
                # Special tag received, we switch the attractor
                lang_inputs += [self.m_tag_symbol]
                seen = 1.0
                seen_pos = count
            else:
                # Choose another tag at random
                sym_pos = np.random.randint(0, len(self.m_other_symbols))
                lang_inputs += [self.m_other_symbols[sym_pos]]
            # endif

            # The corresponding output
            lang_outputs += [[seen]]

            # If we reached the end, switch back
            if seen_pos != -1 and self.m_memory_length != -1 and count - seen_pos >= self.m_memory_length - 1:
                seen_pos = -1
                if self._switch_back_symbol is not None:
                    lang_inputs[-1] = self._switch_back_symbol
                seen = 0.0
            elif seen_pos != -1 and self.m_memory_length != -1:
                seen -= mem_step
            # endif

            # Next element
            count += 1

        # end for

        return lang_inputs, lang_outputs
    # end generate

    # Generate the whole data set.
    def generate_data_set(self, n_samples=10, sample_length=1000):
        """
        Generate the whole data set.
        :param n_samples: Number of samples to generate.
        :param sample_length: The samples' length.
        :return: A data set with n_samples of length sample_length.
        """

        # Inputs / outputs
        lang_inputs = []
        lang_outputs = []

        # For each samples
        for n_sample in range(n_samples):

            # Sparse sample or not
            if np.random.random_sample() <= self.m_sparsity:
                inp, out = self.generate(sample_length, [sample_length + 10])
            else:
                inp, out = self.generate(sample_length, [np.random.randint(0, sample_length - self.m_memory_length * 1.5)])

            # Add
            lang_inputs += [inp]
            lang_outputs += [out]

        # end for

        return np.array(lang_inputs), np.array(lang_outputs)
    # end generate_data_set

# end RCNLPSwitchingAttractorLanguage
