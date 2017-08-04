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
import os
from shutil import copyfile
import logging

#########################################################################
# Experience settings
#########################################################################

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word prediction with Echo State Network and one-hot vector")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--output", type=str, help="Output directory", required=True)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="RCNLP")

    # For each author path
    index = 0
    for author_file in os.listdir(args.dataset):
        # Author path
        author_path = os.path.join(args.dataset, author_file)

        # For each author's text
        for author_text in os.listdir(author_path):
            # File path
            file_path = os.path.join(author_path, author_text)

            # Destination path
            destination_path = os.path.join(args.output, str(index) + ".txt")

            # Copy file
            copyfile(file_path, destination_path)

            # Increment index
            index += 1
        # end for
    # end for

# end if
