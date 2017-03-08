#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 19.04.2015 17:59:05
# Lieu : Nyon, Suisse
#
# Création d'un langage où l'on demande au système de se souvenir d'un
# symbole précédemment présenté à son entrée.
#
###########################################################################

import numpy as np

#
# CLASS MNISTImporter
# Classe permettant l'importation des digits MNIST, leur traitement, leur conversion en
# série temporelle et de multiple transformations avant de le passer au réservoir
#
class SymbolMemoryGenerator:
    
    ##############################################################
    # Constructeur
    ##############################################################
    """def __init__(self, n_symbols, memory_shiffting):
        "" Constructeur, on initialize les variables par défaut ""
        
        self.m_iNbSymbols = n_symbols                               # Number of different symbols
        self.m_iMemoryShiffting = memory_shiffting                  # Number of step we keep the tag symbol
        """
    
    # Génère une chaîne
    def generate(self, n_symbols, memory_shiffting, length):
        
        # Input output
        inputs = np.array([])
        inputs.shape = (0, n_symbols)
        outputs = np.zeros((memory_shiffting, n_symbols))
        
        # For each symbols to remember
        for i in range(length - memory_shiffting):
            symbol = np.zeros(n_symbols)
            symbol[np.random.randint(n_symbols)] = 1
            inputs = np.vstack((inputs, symbol))
            outputs = np.vstack((outputs, symbol))

        # No symbol to remember at the end
        inputs = np.vstack((inputs, np.zeros((memory_shiffting, n_symbols))))
        
        return inputs, outputs
    
    ###############################################################
    # Génère plusieurs chaine
    ###############################################################
    def generateDataset(self, n_symbols, memory_shiffting, n_samples = 10, sample_length = 1000):
        
        # Inputs/ouputs
        inputs = []
        outputs = []
        
        # For each samples
        for n_sample in range(n_samples):
            
            # Generate
            inp, out = self.generate(n_symbols, memory_shiffting, length = sample_length)
            
            # Add
            inputs = inputs + [inp]
            outputs = outputs + [out]
        
        return np.array(inputs), np.array(outputs)
