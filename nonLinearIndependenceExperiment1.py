 # =====================================================================================
 #
 #       Filename:  nonLinearIndependenceExperiment.py
 #
 #    Description:  main src file, contains entry point
 #
 #        Version:  1.0
 #        Created:  Dec 2015
 #       Revision:  none
 #       Compiler:  
 #
 #         Author:  Ahmad Hassan (Ahmadposten)
 #   Organization:  
 #
 # =====================================================================================
''' In this expirement 200 component random vectors of data which are not linearly indep-
endent are tested for linear independence after passing through a non-linearity box which
is a neural network of a single sigmoid layer'''
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
import numpy as np

## Configurations 
nc          = 200 # No of components
n           = 130 # No of vectors
network     = FeedForwardNetwork()
inputLayer  = LinearLayer(nc)
hiddenLayer = SigmoidLayer(400)
results     = [] # Holds 1 if the expirement yielded linearly independend valus
# Generate the other 70% of the data by linear combinations 

#print dataSet
network.addInputModule(inputLayer)
network.addOutputModule(hiddenLayer)

# Connections 
# Weights are randomly initialized
in_to_out = FullConnection(inputLayer, hiddenLayer)
network.addConnection(in_to_out)

network.sortModules()

for x in range(1000):
	dataSet = np.random.rand(n, 200) * 2 - 1 
	for x in range(500-n):
		dataSet = np.append(dataSet, [dataSet[np.random.randint(0,n)] + dataSet[np.random.randint(0,n)]], axis=0)
	activations = map(network.activate, dataSet)
	matrixRank = np.linalg.matrix_rank(np.matrix(activations))
	print matrixRank
	results += [matrixRank == 400]



print results

