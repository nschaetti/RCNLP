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
import logging
import io

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Clean Wikipedia dump")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="RCNLP")

    # For each directory
    for subdirectory in os.listdir(args.dataset):
        # Directory path
        directory_path = os.path.join(args.dataset, subdirectory)

        # Is DIR
        if os.path.isdir(directory_path):
            # Directory path
            logger.info(u"Entering directory {}".format(directory_path))

            # List file
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)

                # Open file
                text_content = io.open(file_path, 'r', encoding='utf-8').read()

                # Replace .\n
                text_content = text_content.replace(u".\n", u". ")
                for i in range(15):
                    text_content = text_content.replace(u"  ", u" ")
                # end for
                for i in range(15):
                    text_content = text_content.replace(u"\n\n", u"\n")
                # end for
                text_content = text_content.replace(u". \n", u". ")
                text_content = text_content.replace(u". #", u".\n#")
                text_content = text_content.replace(u"\n", u". ")
                text_content = text_content.replace(u"#. ", u"#\n")
                text_content = text_content.replace(u". #", u"\n#")

                # Write
                logger.info(u"Writing {}".format(file_path))
                io.open(file_path, 'w', encoding='utf-8').write(text_content)
            # end for
        # end if
    # end for
# end if
