#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 19.04.2015 17:59:05
# Lieu : Nyon, Suisse
# 
# Fichier sous licence GNU GPL
#
###########################################################################

import Oger
from tools.symbol_memory_generator import SymbolMemoryGenerator
from tools.metrics import remembering_rate, equal_output, lucidity, MSSR
from tools.discrete_symbol_node import DiscreteSymbolNode
import mdp
import os
import cPickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np

#########################################################################
#
# Propriétés de l'expérience
#
#########################################################################

# Propriétés du réservoir
rc_SpectralRadius           = 0.99                                              # Spectral radius
rc_Size                     = 500                                                # Taille du réservoir
rc_InputScaling             = 0.5                                               # Dimensionnement des entrées
rc_LeakRate                 = 1.0                                               # Leak rate

# Propriétés du jeu de données
rc_DatasetSize              = 40                                                # Longueur du jeu de données
rc_TrainingLength           = 30                                                # Longueur d'entrainement
rc_TestLength               = rc_DatasetSize - rc_TrainingLength                # Longeur de test
rc_MemoryShiffting          = 4
rc_SampleLength             = 1000
rc_NbSymbols                = np.arange(5,201,10)
rc_NbSamples                = 10

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
    
    # File destination
    if len(sys.argv) >= 2:
        file_output = sys.argv[1]
    else:
        file_output = ""

    # Generate the dataset
    generator = SymbolMemoryGenerator()
    
    # Table of results for each values
    mssres = []

    # Reservoir
    #reservoir = Oger.nodes.LeakyReservoirNode(input_dim = rc_NbSymbols, output_dim = rc_Size, input_scaling = rc_InputScaling, leak_rate = rc_LeakRate)
    #symbol_out = DiscreteSymbolNode(input_dim = rc_NbSymbols, output_dim = rc_NbSymbols)

    # Create the flow
    #flow = mdp.Flow([reservoir, readout, symbol_out], verbose=1)

    # Varying a property
    for nb_symbols in rc_NbSymbols:

        # Multiple sample
        sample = []
        for n in np.arange(0,rc_NbSamples):

            # Reservoir
            readout = Oger.nodes.RidgeRegressionNode()

            reservoir = Oger.nodes.LeakyReservoirNode(input_dim = nb_symbols, output_dim = rc_Size, input_scaling = rc_InputScaling, leak_rate = rc_LeakRate)
            #readout = Oger.nodes.RidgeRegressionNode()
            symbol_out = DiscreteSymbolNode(input_dim = nb_symbols, output_dim = nb_symbols)

            # Create the flow
            flow = mdp.Flow([reservoir, readout, symbol_out], verbose=1)

            # Generating data to learn
            inputs, outputs = generator.generateDataset(n_symbols = nb_symbols, memory_shiffting = rc_MemoryShiffting, sample_length = rc_SampleLength, n_samples = rc_DatasetSize)

            # Training and test
            training_in, training_out   = inputs[0:rc_TrainingLength], outputs[0:rc_TrainingLength]
            test_in, test_out           = inputs[rc_TrainingLength:], outputs[rc_TrainingLength:]
            
            # Reservoir input data
            data = [training_in, zip(training_in, training_out), None]
            
            # Train
            flow.train(data)

            # For each test set
            t = 0
            mssr = []
            for test_inputs in test_in:

                # Give to reservoir
                test_outputs = flow(test_inputs)

                # Remove first outputs that are not linked
                # to a previous symbol entered
                test_outputs = test_outputs[rc_MemoryShiffting:]
                targets = test_out[t][rc_MemoryShiffting:]

                # Print MSSR
                mssr += [MSSR(test_outputs, targets)]
                #print("MSSR : " + str(MSSR(test_outputs, targets) * 100.0))

                # Compare y and y_
                """for i in range(rc_SampleLength - rc_MemoryShiffting):
                    print(str(targets[i]) + " ---- " + str(test_outputs[i]))
                """
                t += 1

            # Add result
            sample += [np.average(mssr) * 100.0]
            #print("MSSR for size " + str(size) + " : " + str(np.average(mssr) * 100.0))

        # Add to all samples
        mssres += [sample]

    # Put into average and std
    mssr_averages = []
    mssr_errors = []
    for mssr_sample in mssres:
        mssr_averages += [np.average(mssr_sample)]
        mssr_errors += [np.std(mssr_sample)]

    # Display the graph
    #plt.plot(mssres)
    #plt.set_title("MSSR vs nb. of symbols")
    plt.errorbar(rc_NbSymbols, mssr_averages, yerr = mssr_errors, fmt='-o')
    #plt.plot(rc_NbSymbols, np.full((len(mssres)), 1.0 / float(rc_NbSymbols) * 100.0))
    bottom_values = []
    for nb_symbols in rc_NbSymbols:
        bottom_values += [1.0 / float(nb_symbols) * 100.0]
    plt.plot(rc_NbSymbols, bottom_values)
    plt.show()

    # Save to file
    if file_output != "":
        f = open(file_output, 'w')
        f.write("Averages : ")
        for a in mssr_averages:
            f.write(str(a) + ",")
        f.write("\n")
        f.write("Errors : ")
        for e in mssr_errors:
            f.write(str(e) + ",")
        f.close()
