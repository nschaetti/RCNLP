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
import argparse
from core.converters.RCNLPPosConverter import RCNLPPosConverter


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship attribution with Part-Of-Speech to Echo State Network")

    # Argument
    parser.add_argument("--file", type=str, help="Input text file")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    args = parser.parse_args()

    converter = RCNLPPosConverter()
    doc_array = converter(io.open(args.file, 'r').read())
    print(doc_array)
    print(doc_array.shape)
    print(doc_array[0])
    RCNLPPosConverter.display_representations(doc_array)
    data_set = RCNLPPosConverter.generate_data_set_inputs(doc_array, 2, 0)
    print(data_set)

# end if