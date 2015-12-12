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
n           = 30 # No of vectors
network     = FeedForwardNetwork()
network2    = FeedForwardNetwork()
inputLayer1 = LinearLayer(nc)
hiddenLayer1= SigmoidLayer(100)
results1    = [] # Holds 1 if the expirement yielded linearly independend valus
results2    = [] 
# The 200 output case 

inputLayer2  = LinearLayer(nc)
hiddenLayer2 = SigmoidLayer(200)

# Generate the other 70% of the data by linear combinations 

#print dataSet
network.addInputModule(inputLayer1)
network.addOutputModule(hiddenLayer1)

network2.addInputModule(inputLayer2)
network2.addOutputModule(hiddenLayer2)

# Connections 
# Weights are randomly initialized
in_to_out  = FullConnection(inputLayer1, hiddenLayer1)
in_to_out2 = FullConnection(inputLayer2, hiddenLayer2) 
network.addConnection(in_to_out)
network2.addConnection(in_to_out2)

network.sortModules()
network2.sortModules()
## Calculating activations for all of the dataset 
## Repeat 999 times 

for x in range(2):
	dataSet = np.random.rand(30, 200) * 2 - 1 
	for x in range(100):
		dataSet = np.append(dataSet, [dataSet[np.random.randint(0,30)] + dataSet[np.random.randint(0,30)]], axis=0)

	activations1 = map(network.activate, dataSet)
	
	dataSet = np.random.rand(65, 200) * 2 - 1 
	for x in range(145):
		dataSet = np.append(dataSet, [dataSet[np.random.randint(0,65)] + dataSet[np.random.randint(0,65)]], axis=0)
	activations2 = map(network2.activate, dataSet)

	matrixRank1 = np.linalg.matrix_rank(np.matrix(activations1))
	matrixRank2 =  np.linalg.matrix_rank(np.matrix(activations2))
	results1 += [matrixRank1 == 100]
	results2 += [matrixRank2 == 200]



print results1

print results2

