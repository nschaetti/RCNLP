#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.TextClassifier.py
# Description : Text classifier abstract class.
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

# Import
import numpy as np


# Text classifier
class TextClassifier(object):
    """
    Text classifier
    """

    # Text classifier
    _training_finalized = False

    # Constructor
    def __init__(self, classes, lang='en_core_web_md'):
        """
        Constructor
        :param classes: Classes
        :param lang: Spacy language
        """
        # Properties
        self._classes = classes
        self._n_classes = len(classes)
        self._lang = lang
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Train the model
    def train(self, x, y, verbose=False):
        """
        Train the model
        :param x: Example's inputs.
        :param y: Example's outputs.
        :param verbose: Verbosity
        """
        pass
    # end train

    # Finalize model training
    def finalize(self, verbose=False):
        """
        Finalize model training
        """
        if not self._training_finalized:
            self._finalize_training(verbose)
            self._training_finalized = True
        # end if
    # end finalize

    # Predict the class
    def predict(self, x):
        """
        Predict class of a text file
        :param x: Sample
        :return: Predicted class and classes probabilities
        """
        return self._classify(x)
    # end predict

    # Reset the classifier
    def reset(self):
        """
        Reset the classifier
        """
        self._reset_model()
        self._training_finalized = False
    # end reset

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
        if not self._training_finalized:
            self._finalize_training()
            self._training_finalized = True
        # end if

        # Classify the document
        return self._classify(x)
    # end __class__

    # To str
    def __str__(self):
        """
        To string
        :return:
        """
        pass
    # end __str__

    ##############################################
    # Private
    ##############################################

    # Filter token
    def _filter_token(self, word):
        """
        Filter token
        :param token:
        :return:
        """
        word_text = word.text
        word_text = word_text.replace(u"\n", u"")
        word_text = word_text.replace(u"\t", u"")
        word_text = word_text.replace(u"\r", u"")
        if len(word_text) > 0:
            word_vector = word.vector
            if np.average(word_vector) != 0:
                return True, word_text
            # end if
        # end if
        return False, ""
    # end if

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
    def _finalize_training(self, verbose=False):
        """
        Finalize training.
        :param verbose: Verbosity
        """
        pass
    # end _finalize_training

    # Reset the model
    def _reset_model(self):
        """
        Reset the model
        """
        pass
    # end _reset_model

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
