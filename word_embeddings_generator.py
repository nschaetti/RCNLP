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

import argparse
from core.embeddings.Word2Vec import Word2Vec
from core.embeddings.WordPredictionDataset import WordPredictionDataset


####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word Embeddings (Vector) generator")

    # Argument
    args = parser.parse_args()

    # Word2Vec
    word2vec = Word2Vec(dim=10, mapper='sparse', sparsity=0.2)

    # Wordprediction dataset generator
    dataset = WordPredictionDataset(word2vec=word2vec)

    # Add examples
    dataset.add(u"Hello, what is your name?")
    dataset.add(u"When do you want to go to Disneyland?")
    dataset.add(u"Hi! What's up?")

    # Data
    data = dataset.get_dataset()

    print(type(data))
    print(len(data))
    print(type(data[0]))
    print(len(data[0]))
    print(data[0][0])
    print(data[0][1])

    print(word2vec[u'DisneyLand'])
# end if
