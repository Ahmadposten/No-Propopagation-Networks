import binascii
import numpy as np 
import time
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from matplotlib import pyplot as plt
from pybrain.utilities  import percentError
import os.path
from pybrain.datasets            import ClassificationDataSet
import random
import math

def readData(path):
	f = open(path, 'r')
	magicNo    = int(binascii.hexlify(f.read(4)), 16)
	noOfImages = int(binascii.hexlify(f.read(4)), 16)
	nRows      = int(binascii.hexlify(f.read(4)), 16)
	nCols      = int(binascii.hexlify(f.read(4)), 16)

	images = []
	image = np.zeros((nRows, nCols))

	for im in range(noOfImages):
		for i in range(nRows):
			for j in range(nCols):
				image[i][j] = int(binascii.hexlify(f.read(1)), 16)

		images += [image]
	images = np.array(images)
	f.close()
	return images

def readLabel(path):
	f = open(path, 'r')
	magicNo = int(binascii.hexlify(f.read(4)), 16)
	noItems = int(binascii.hexlify(f.read(4)), 16)
	labels = []
	for x in range(noItems):
		labels += [int(binascii.hexlify(f.read(1)), 16)]
	return np.array(labels)

def generateRandomInices(r, c, p, t):
	l = []
	while len(l) < p:
		randomIndex = (random.sample(range(r),1)[0], random.sample(range(c),1)[0])
		if randomIndex not in t and randomIndex not in l:
			l += [randomIndex]
	return l
		
def addNoise(dataSet, percentage, takenIndices = []):
	[rows, cols]   = np.shape(dataSet[0])
	pixelsAffected = rows * cols * percentage / 100 
	indices = generateRandomInices(rows, cols, pixelsAffected, takenIndices)
	for im in dataSet:
		for (i,j) in indices:
			im[i][j] = np.round(random.random()) * im[i][j]

	takenIndices += indices
	return (dataSet, takenIndices)
	


def normalizeData(dataset):
	pass

def updateWeights(weights, error, inputvector):
	add =   .001* inputvector * error 
	newweights = []
	for i,x in enumerate(weights):
		newweights += [x + add[i]]
	return newweights
def lmsTrain(network, dataset, targets, itterations):
	networkParams = network.params
	lastlayerParams = networkParams[(len(networkParams) - (10*784)):]
	weights = [] 
	accumlator = []
	errorsq = 1
	for x in lastlayerParams:
		accumlator += [x]
		if(len(accumlator) == 784):
			weights += [accumlator]
			accumlator = []
	
	lasLayerParams = np.array(weights);
	errors = []
	for i in range(itterations):
		jjjj = []
		for j,x in enumerate(dataset):
			y = network.activate(x)
			error = y - targets[j]
			jjjj += [sum(error * 2)]
			newweights = []
			for (b,w) in enumerate(weights):
				w = updateWeights(w, error[b], x)
				newweights += [w]
			weights = newweights
			hhh = np.array(weights).flatten()
			networkParams[(len(networkParams) - (10*784)):] = hhh 
			network._setParameters(p=networkParams)
		errors += [sum(jjjj) / len(jjjj)]
		print "Itteration : " + str(i+1)
		print "Error: " + str( sum(jjjj) / len(jjjj))
	

	print "done training for ", itterations, " itterations"
	"Errors ", errors
	return np.array(weights)

			

	
def getDistinctOf(indices):
	x = []
	for i in indices:
		if i not in x:
			x+=[i]
	return (len(x), len(indices));

if __name__ == "__main__":
	if not os.path.exists('dataset.npy') or not os.path.exists('datalabels.npy'):
		dataSet						     = readData('train-images-idx3-ubyte')
		labels							 = readLabel('train-labels-idx1-ubyte')
		testSet							 = readData('t10k-images-idx3-ubyte')
		testLabels						 = readLabel('t10k-labels-idx1-ubyte')

		[noise5Data, takenIndices]       = addNoise(dataSet,	 5)
		[noise10Data, takenIndices]      = addNoise(noise5Data,  5, takenIndices)
		[noise15Data, takenIndices]      = addNoise(noise10Data, 5, takenIndices)
		[noise20Data, takenIndices]      = addNoise(noise15Data, 5, takenIndices)
		[noise25Data, takenIndices]      = addNoise(noise20Data, 5, takenIndices)
		[noise30Data, takenIndices]      = addNoise(noise25Data, 5, takenIndices)

		print getDistinctOf(takenIndices);

		
		#Vectorize dataSet
		

		dataSet			 = np.array(map(lambda a : a.flatten().tolist(), dataSet))
		noise5Data		 = map(lambda a : a.flatten(), noise5Data)
		noise10Data		 = map(lambda a : a.flatten(), noise10Data)
		noise15Data		 = map(lambda a : a.flatten(), noise15Data)
		noise20Data		 = map(lambda a : a.flatten(), noise20Data)
		noise25Data		 = map(lambda a : a.flatten(), noise25Data)
		noise30Data		 = map(lambda a : a.flatten(), noise30Data)

		[noise5TestSet,t]      = addNoise(testSet, 5)
		[noise10TestSet,t]     = addNoise(noise5TestSet, 5, t)
		[noise15TestSet,t]     = addNoise(noise10TestSet, 5, t)
		[noise20TestSet,t]     = addNoise(noise15TestSet, 5, t)
		[noise25TestSet,t]     = addNoise(noise20TestSet, 5, t)
		[noise30TestSet,t]     = addNoise(noise25TestSet, 5, t)
		testSet			 = map(lambda a : a.flatten(), testSet)
		noise5TestSet	     = map(lambda a : a.flatten(), noise5TestSet)
		noise10TestSet	     = map(lambda a : a.flatten(), noise10TestSet)
		noise15TestSet	     = map(lambda a : a.flatten(), noise15TestSet)
		noise20TestSet	     = map(lambda a : a.flatten(), noise20TestSet)
		noise25TestSet	     = map(lambda a : a.flatten(), noise25TestSet)
		noise30TestSet	     = map(lambda a : a.flatten(), noise30TestSet)

		np.save('dataset', dataSet)
		np.save('dataset5noise', noise5Data)
		np.save('dataset10noise', noise10Data)
		np.save('dataset15noise', noise15Data)
		np.save('dataset20noise', noise20Data)
		np.save('dataset25noise', noise25Data)
		np.save('dataset30noise', noise30Data)
		np.save('datalabels', labels)
		np.save('testlabels', testLabels)
		np.save('testdata', testSet)
		np.save('test5noise', noise5TestSet)
		np.save('test10noise', noise10TestSet)
		np.save('test15noise', noise15TestSet)
		np.save('test20noise', noise20TestSet)
		np.save('test25noise', noise25TestSet)
		np.save('test30noise', noise30TestSet)

		print "stuff saved"

	else:
		dataSet     = np.load('dataset.npy')[0:2000]
		noise5Data  = np.load('dataset5noise.npy')[0:2000]
		noise10Data = np.load('dataset10noise.npy')[0:2000]
		noise15Data = np.load('dataset15noise.npy')[0:2000]
		noise20Data = np.load('dataset20noise.npy')[0:2000]
		noise25Data = np.load('dataset25noise.npy')[0:2000]
		noise30Data = np.load('dataset30noise.npy')[0:2000]
		labels      = np.load('datalabels.npy')[0:2000]
		testSet     = np.load('testdata.npy')[0:5000]
		test5noise  = np.load('test5noise.npy')[0:5000]
		test10noise = np.load('test10noise.npy')[0:5000]
		test15noise = np.load('test15noise.npy')[0:5000]
		test20noise = np.load('test20noise.npy')[0:5000]
		test25noise = np.load('test25noise.npy')[0:5000]
		test30noise = np.load('test30noise.npy')[0:5000]
		testLabels  = np.load('testlabels.npy')[0:5000]
		print "data loaded"

	
	inputComponents  = np.shape(dataSet)[1]
	outputComponents = 10 
	network = FeedForwardNetwork()
	
	inputLayer  = LinearLayer(inputComponents,name='input')
	hiddenLayer = SigmoidLayer(inputComponents, name='hidden')
	outputLayer = SigmoidLayer(outputComponents,name='out')

	data = ClassificationDataSet(inputComponents, 1, nb_classes = 10)
	in_hidden = FullConnection(inputLayer, hiddenLayer)
	hidden_out = FullConnection(hiddenLayer, outputLayer)

	network.addInputModule(inputLayer)
	network.addModule(hiddenLayer)
	network.addOutputModule(outputLayer)

	network.addConnection(in_hidden)
	network.addConnection(hidden_out)
	network.sortModules()
	print network.params
	x = network.params 

	print "fuck", network.connections
	targets = []

	for h in labels:
		j = [0,0,0,0,0,0,0,0,0,0]
		j[h] = 1
		targets +=  [j]

	newParams = lmsTrain(network, dataSet, targets, 20)
	newParams = newParams.flatten()
	x[(len(x) - (784 * 10)):] = newParams
	network._setParameters(p=x)
	activations = np.zeros(10)
	results = []

	for x in dataSet:
		activations = np.zeros(10)
		r = network.activate(x)
		activations[np.argmax(r)] = 1
		results += [1]
	
	testTargets = []
	for x in testLabels:
		h = np.zeros(10)
		h[x] = 1
		testTargets += [h]
	

	trainingErrors = []

	for i,x in enumerate(results):
		if x != targets[i]:
			trainingErrors += [1]
	print trainingErrors
	trainingError = len(trainingErrors) / len(results) * 100

	
	print "Training error", trainingError

	testResults = []
	for x in testSet:
		activation = np.zeros(10)
		activations[np.argmax(network.activate(x))] = 1
		testResults += [activations]
	
	testErrors = []
	for i,x in enumerate(testSet):
		if x != testTargets[i]:
			testErrors += [1]
	
	testError = len(testErrors) / len(testSet) * 100 

	print "Test Error is", testError 
	


