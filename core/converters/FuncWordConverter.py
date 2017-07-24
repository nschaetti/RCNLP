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

import numpy as np
import spacy
from RCNLPConverter import RCNLPConverter


class FuncWordConverter(RCNLPConverter):
    """
    Convert text to function word symbols.
    """

    # Constructor
    def __init__(self, lang='en', tag_to_symbol=None, resize=-1, pca_model=None, fill_in=False):
        """
        Constructor
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        super(FuncWordConverter, self).__init__(lang, tag_to_symbol, resize, pca_model)
        self._fill_in = fill_in
    # end __init__

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A tag list.
        """
        return [u"a", u"about", u"above", u"after", u"after", u"again", u"against", u"ago", u"ahead",
                u"all",
                u"almost", u"along", u"already", u"also", u"although", u"always", u"am", u"among", u"an",
                u"and", u"any", u"are", u"aren't", u"around", u"as", u"at", u"away", u"backward",
                u"backwards", u"be", u"because", u"before", u"behind", u"below", u"beneath", u"beside",
                u"between", u"both", u"but", u"by", u"can", u"cannot", u"can't", u"cause", u"'cos",
                u"could",
                u"couldn't", u"'d", u"despite", u"did", u"didn't", u"do", u"does", u"doesn't", u"don't",
                u"down", u"during", u"each", u"either", u"even", u"ever", u"every", u"except", u"for",
                u"forward", u"from", u"had", u"hadn't", u"has", u"hasn't", u"have", u"haven't", u"he",
                u"her", u"here", u"hers", u"herself", u"him", u"himself", u"his", u"how", u"however",
                u"I",
                u"if", u"in", u"inside", u"inspite", u"instead", u"into", u"is", u"isn't", u"it", u"its",
                u"itself", u"just", u"'ll", u"least", u"less", u"like", u"'m", u"many", u"may",
                u"mayn't",
                u"me", u"might", u"mightn't", u"mine", u"more", u"most", u"much", u"must", u"mustn't",
                u"my", u"myself", u"near", u"need", u"needn't", u"needs", u"neither", u"never", u"no",
                u"none", u"nor", u"not", u"now", u"of", u"off", u"often", u"on", u"once", u"only",
                u"onto",
                u"or", u"ought", u"oughtn't", u"our", u"ours", u"ourselves", u"out", u"outside", u"over",
                u"past", u"perhaps", u"quite", u"'re", u"rather", u"'s", u"seldom", u"several", u"shall",
                u"shan't", u"she", u"should", u"shouldn't", u"since", u"so", u"some", u"sometimes",
                u"soon",
                u"than", u"that", u"the", u"their", u"theirs", u"them", u"themselves", u"then", u"there",
                u"therefore", u"these", u"they", u"this", u"those", u"though", u"through", u"thus",
                u"till",
                u"to", u"together", u"too", u"towards", u"under", u"unless", u"until", u"up", u"upon",
                u"us", u"used", u"usedn't", u"usen't", u"usually", u"'ve", u"very", u"was", u"wasn't",
                u"we", u"well", u"were", u"weren't", u"what", u"when", u"where", u"whether", u"which",
                u"while", u"who", u"whom", u"whose", u"why", u"will", u"with", u"without", u"won't",
                u"would", u"wouldn't", u"yet", u"you", u"your", u"yours", u"yourself", u"yourselves"]
    # end get_tags

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return: The input size.
        """
        return len(self.get_tags())
    # end if

    # Convert a string to a ESN input
    def __call__(self, text, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param text: The text to convert.
        :return: An numpy array of inputs.
        """

        # Load language model
        nlp = spacy.load(self._lang)

        # Process text
        doc = nlp(text)

        # Resulting numpy array
        doc_array = np.array([])

        # Null symbol
        null_symbol = np.zeros((1, len(self.get_tags())))

        # For each words
        init = False
        for index, word in enumerate(doc):
            if word not in word_exclude:
                sym = self.tag_to_symbol(word.text)
                if sym is None and self._fill_in:
                    sym = null_symbol
                # end if
                if sym is not None:
                    if not init:
                        doc_array = sym
                        init = True
                    else:
                        doc_array = np.vstack((doc_array, sym))
                    # end if
                # end if
            # end if
        # end for

        return self.reduce(doc_array)
    # end convert

# end RCNLPConverter
