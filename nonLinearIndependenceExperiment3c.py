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

# A neural network of a single hidden layer of 100 neurons and an output layer of 100 neurons is used 
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
import numpy as np

## Configurations 
nc          = 200 # No of components
n           = 130 # No of vectors
network     = FeedForwardNetwork()
inputLayer  = LinearLayer(nc)
hiddenLayer = SigmoidLayer(100)
outputLayer = SigmoidLayer(400)
results		= []

network.addInputModule(inputLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outputLayer)
# Connections 
# Weights are randomly initialized
in_to_out  = FullConnection(inputLayer, hiddenLayer)
h_to_out   = FullConnection(hiddenLayer, outputLayer)
network.addConnection(in_to_out)
network.addConnection(h_to_out)

network.sortModules()
## Calculating activations for all of the dataset 
for x in range(1000):
	dataSet = np.random.rand(n, 200) * 2 - 1 
	for x in range(400-n):
		dataSet = np.append(dataSet, [dataSet[np.random.randint(0,n)] + dataSet[np.random.randint(0,n)]], axis=0)
	activations = map(network.activate, dataSet)
	matrixRank  = np.linalg.matrix_rank(np.matrix(activations))
	results += [matrixRank == 400]


print results
