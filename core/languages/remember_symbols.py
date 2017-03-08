#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 19.04.2015 17:59:05
# Lieu : Nyon, Suisse
#
# Création d'un langage où l'on demande au système si il a déjà vu un 
# symbole.
#
###########################################################################

import numpy as np

#
# CLASS RememberSymbols
# Classe permettant l'importation des digits MNIST, leur traitement, leur conversion en
# série temporelle et de multiple transformations avant de le passer au réservoir
#
class RememberSymbols:
	
	##############################################################
	# Constructeur
	##############################################################
	def __init__(self, n_symbols):
		""" Constructeur, on initialize les variables par défaut """
		
		self.m_sTagSymbol = tag_symbol						# Symbol to remember
		self.m_aOtherSymbols = other_symbols				# Other symbols taken randomly
		self.m_iMemoryLength = memory_length				# Number of step we keep the tag symbol
		self.m_bSloppingMemory = slopping_memory			# Is the memory slopping?
	
	###############################################################
	# Génère une chaîne
	###############################################################
	def generate(self, length, tag_places):
		
		# Input output
		inputs = []
		outputs = []
		
		# Memory step
		if self.m_bSloppingMemory:
			mem_step = 1.0 / self.m_iMemoryLength
		else:
			mem_step = 0.0
		
		# Generate each symbols
		count = 0
		seen = 0
		seen_pos = -1
		for i in range(length):
			
			if count in tag_places:
				inputs = inputs + [self.m_sTagSymbol]
				seen = 1.0
				seen_pos = count
			else:
				sym_pos = np.random.randint(0, len(self.m_aOtherSymbols))
				inputs = inputs + [self.m_aOtherSymbols[sym_pos]]
			
			outputs = outputs + [[seen]]
			
			if seen_pos != -1 and self.m_iMemoryLength != -1 and count - seen_pos >= self.m_iMemoryLength - 1:
				seen_pos = -1
				seen = 0.0
			elif seen_pos != -1 and self.m_iMemoryLength != -1:
				seen -= mem_step
			
			count += 1
		
		return (inputs, outputs)
	
	###############################################################
	# Génère plusieurs chaine
	###############################################################
	def generateDataset(self, n_samples = 10, sample_length = 1000, sparsity = 0.1):
		
		# Inputs/ouputs
		inputs = []
		outputs = []
		
		# For each samples
		for n_sample in range(n_samples):
			
			# Sparse sample or not
			if np.random.random_sample() <= sparsity:
				inp, out = self.generate(sample_length, [sample_length + 10])
			else:
				inp, out = self.generate(sample_length, [np.random.randint(0, sample_length - self.m_iMemoryLength*1.5)])
			
			# Add
			inputs = inputs + [inp]
			outputs = outputs + [out]
		
		return np.array(inputs), np.array(outputs)
