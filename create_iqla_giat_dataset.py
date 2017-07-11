#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_iqla_giat_dataset.py
# Description : Create the IQLA-GIAT dataset.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 10.07.2017 13.51
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
import random
import string
import json
import logging
import codecs

####################################################
# Main function
####################################################

# Main part
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Create the IQLA-GIAT dataset from brut texts.")

    # Argument
    parser.add_argument("--texts", type=str, help="Text directory", required=True)
    parser.add_argument("--output", type=str, help="Output directory", required=True)
    args = parser.parse_args()

    # Init data structure
    authors_data = dict()
    texts_data = dict()

    # For each author
    for author_name in os.listdir(args.texts):
        # Author's path
        author_path = os.path.join(args.texts, author_name)

        # Is dir
        if os.path.isdir(author_path):
            # Log
            logging.info("Author {}".format(author_name))

            # Init author's data
            authors_data[author_name] = list()

            # For each author's novels
            for novel_file in os.listdir(author_path):
                # Novel's path
                novel_path = os.path.join(author_path, novel_file)

                # Generate a random code
                text_code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                destination_path = os.path.join(args.output, text_code + ".txt")

                # Add to texts info
                authors_data[author_name].append(text_code)
                texts_data[text_code] = author_name

                # Open text file
                novel_text = codecs.open(novel_path, 'r', encoding='utf-8').read()

                # Open destination file
                df = codecs.open(destination_path, 'w', encoding='utf-8')

                # For each line
                for line in novel_text.split(u'\n'):
                    if u"<DOCNO>" not in line:
                        df.write(line + u"\n")
                    # end if
                # end for
            # end for
        # end if
    # end for

    # Write author info
    logging.info("Writing author data file...")
    with open(os.path.join(args.output, "authors.json"), 'w') as f:
        json.dump(authors_data, f, sort_keys=True, indent=4)
    # end with

    # Write texts info
    logging.info("Writing texts data file...")
    with open(os.path.join(args.output, "texts.json"), 'w') as f:
        json.dump(texts_data, f, sort_keys=True, indent=4)
    # end with

# end if