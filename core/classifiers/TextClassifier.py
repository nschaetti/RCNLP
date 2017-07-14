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

# Import packages
import numpy as np
import Oger
import math
import mdp


# TextClassifier
class TextClassifier(object):

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes: Classes
        """
        # Properties
        self._classes = classes
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
    # Override
    ##############################################

    # Class the classifier
    def __call__(self, x):
        """
        Class a text document.
        :param x: Document's text.
        :return: A tuple with found class and values per classes.
        """
        # Finalize training
        self._finalize_training()

        # Classify the document
        return self._classify(x)
    # end __class__

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

    # Transform int to class name
    def _int_to_class(self, index):
        """
        Transform index to class name.
        :param class_index: Class index.
        :return: Class name.
        """
        return self._classes[index]
    # end _int_to_class

    # Transform class name to int
    def _class_to_int(self, class_name):
        """
        Transform class name to int
        :param class_name: Class name
        :return: Integer
        """
        for index, name in self._classes:
            if name == class_name:
                return index
            # end if
        # end for
        return -1
    # end class_to_int

# end TextClassifier
