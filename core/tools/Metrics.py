#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
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
#

# Import package
import math
import logging
import numpy as np


class Metrics:

    # Constructor
    def __init__(self):
        pass

    @staticmethod
    def equal_rate(outputs, targets):
        """
        Calculate the percentage of equal elements
        between outputs and targets.
        :param self: Class
        :param outputs: The output of the memory system.
        :param targets: The target output signal to replicate.
        :return: A equal rate between 0 and 100.
        """
        pos = 0
        right_count = 0.0
        for target in targets:
            if target == outputs[pos]:
                right_count += 1.0
            # endif
            pos += 1
        # endfor

        return right_count / float(len(outputs)) * 100.0
    # end equal_rate

    # Remembering rate : how many step is remembered correctly.
    @staticmethod
    def remembering_rate(predicted, observed, threshold=1.0, use_abs=True):
        """
        Remembering rate : how many step is remembered correctly.
        The higher the better.
        :param self: Class
        :param predicted: The output of the memory system.
        :param observed: The target output signal to replicate.
        :param threshold: The threshold to consider the output as switched/remembered
        :param use_abs: Take absolute value?
        :return: A remembering rate between 0 and 100.
        """
        pos = 0
        mem_count = 0.0
        right_count = 0.0
        outputs = []
        for obs in observed:
            if obs == 1.0:
                mem_count += 1.0
                if use_abs and abs(predicted[pos]) >= threshold:
                    right_count += 1.0
                elif not use_abs and predicted[pos] >= threshold:
                    right_count += 1.0
                # endif
            # endif

            if use_abs and abs(predicted[pos]) >= threshold:
                outputs += [1.0]
            elif not use_abs and predicted[pos] >= threshold:
                outputs += [1.0]
            else:
                outputs += [0.0]

            pos += 1
        # endfor

        return float((right_count / mem_count) * 100.0), outputs
    # end remembering_rate

    # Lucidity
    @staticmethod
    def lucidity(predicted, observed, threshold=1.0, use_abs=True):
        """
        Lucidity is the percentage of the time the memory
        system remember having seen the entry even if there
        was no entry. The higher the better.
        :param self: Class
        :param predicted: The output of the memory system.
        :param observed: The target output signal to replicate.
        :param threshold: The threshold to consider the output as switched/remembered
        :param use_abs: Take absolute value?
        :return:
        """
        pos = 0
        mem_count = 0.0
        right_count = 0.0
        for obs in observed:
            if obs == 0.0:
                mem_count += 1.0
                if use_abs and abs(predicted[pos]) < threshold:
                    right_count += 1.0
                elif not use_abs and predicted[pos] < threshold:
                    right_count += 1.0
                # endif
            # endif
            pos += 1
        # end for

        return float(right_count / mem_count * 100.0)
    # end lucidity

    # Symbol recovery rate
    @staticmethod
    def symbol_recovery_rate(mems, targets):
        """
        The number of symbol correctly recovered
        from the memory system. Higher is better.
        :param self: Class
        :param mems: The memory symbol output from the memory system.
        :param targets: The true symbol output.
        :return: The symbol recovery rate between 0 and 100.
        """

        # Check that the two signals have
        # the same number of dimensions.
        if mems.shape != targets.shape:
            return -1.0

        pos = 0
        add = 0
        for target in targets:
            if (target == mems[pos]).sum() == len(target):
                add += 1.0
            # endif
            pos += 1
        # endfor

        return add / float(len(targets))
    # end symbol_recovery_rate

    # L2 distance between two signals.
    @staticmethod
    def average_distance(outputs, targets):
        """
        The average euclidian L2 distance between the
        mutlidimensional outputs. Lower is better.
        :param self: Class
        :param outputs: The output of the system.
        :param targets: The target signal to learn.
        :return: An average distance (float).
        """

        # Check that the two signals have
        # the same number of dimensions.
        if outputs.shape != targets.shape:
            return -1.0

        out = 0
        total = 0.0
        for target in targets:
            count = 0.0
            for i in range(len(target)):
                count += math.pow(target[i] - outputs[out][i], 2)
            # endfor
            total += math.sqrt(count)
            out += 1
        # endfor

        return total / float(len(targets))
    # end average_distance

    # Classification success rate
    @staticmethod
    def success_rate(classifier, test_set, verbose=False, debug=False):
        """
        Classification success rate
        :param classifier: Classifier to test
        :param test_set : Test set
        :param verbose: Verbosity
        :param debug: Display debug informations?
        :return: Classification success rate
        """
        # Counters
        success = 0.0
        count = 0.0

        # For each elements
        for xy in test_set:
            prediction, probs = classifier(xy[0])
            if verbose:
                print(probs)
            # end if
            if prediction == xy[1]:
                success += 1.0
            elif debug:
                classifier.debug()
            # end if

            # Debug
            logging.getLogger("RCNLP").debug(u"Author {}, prediction {}".format(xy[1], prediction))

            # Count
            count += 1.0
        # end for

        return success / count * 100.0
    # end success_rate

    # Relatedness : evaluate word embeddings
    @staticmethod
    def relatedness(word_similarity, word_embeddings, distance_measure='euclidian'):
        """
        Relatedness : evaluate word embeddings
        :param word_similarity:
        :param word_embeddings:
        :param distance_measure:
        :return:
        """
        # Stock correlation factors
        correlations = np.array([])

        # Word count
        word_count = 0

        # For each word in dataset
        for word1, sims in word_similarity:
            if len(sims) > 1:
                if word1 in word_embeddings.words():
                    sim_vector1 = np.array([])
                    sim_vector2 = np.array([])
                    for word2, similarity in sims:
                        if word2 in word_embeddings.words():
                            sim1 = similarity
                            sim2 = word_embeddings.similarity(word1, word2, distance_measure)
                            if sim1 is not None and sim2 is not None:
                                sim_vector1 = np.append(sim_vector1, sim1)
                                sim_vector2 = np.append(sim_vector2, sim2)
                            # end if
                        # end if
                    # end for

                    # If more than one word
                    if sim_vector1.shape[0] > 1:
                        # Compute correlation
                        corfact = np.corrcoef(sim_vector1, sim_vector2)[0, 1]

                        if not np.isnan(corfact):
                            # Add
                            correlations = np.append(correlations, corfact)

                            # Word count
                            word_count += sim_vector1.shape[0]
                        # end if
                    # end if
                # end if
            # end if
        # end for

        return np.average(correlations), word_count
    # end relatedness

# end Metrics
