#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.KLDivergenceTextClassifier.py
# Description : KL divergence text classifier.
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

# Imports
from .TextClassifier import TextClassifier


# KL divergence text classifier
class KLDivergenceTextClassifier(TextClassifier):

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes: Classes
        """
        # Class super class
        super(KLDivergenceTextClassifier, self).__init__(classes=classes)
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Train the model
    def train(self, x, y):
        """
        Train
        :param x: Example's inputs.
        :param y: Example's outputs.
        """
        pass
    # end train

    ##############################################
    # Private
    ##############################################

    # Classify a document
    def _classify(self, x):
        """
        Classify a document.
        :param x: Document's text.
        :return: A tuple with found class and values per classes.
        """
        pass
    # end _classify

    # Finalize the training
    def _finalize_training(self):
        """
        Finalize training.
        """
        pass
    # end _finalize_training

# end KLDivergenceTextClassifier
