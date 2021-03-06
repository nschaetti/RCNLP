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

import io
import json


class PAN16AuthorDiarizationLoader(object):

    # Constructor
    def __init__(self):
        pass
    # end __init__

    # Call
    def __call__(self, truth_file, text_file):
        """

        :param truth_file:
        :param text_file:
        :return:
        """

        # Load JSON truth
        truth = json.load(io.open(truth_file, 'r'))

        # Load text
        text = io.open(text_file, 'r').read()

        # Foreach author
        data_set = []
        for author in truth['authors']:
            texts = []
            # For each extend
            for extent in author:
                texts += [text[extent['from']:extent['to']]]
            # end for
            data_set += [texts]
        # end for

        return data_set
    # end __call__

# end PAN16AuthorDiarizationLoader