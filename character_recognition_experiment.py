# =====================================================================================
 #
 #       Filename:  current_recognition_experiment.py
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

import binascii
import numpy as np 
import time
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from matplotlib import pyplot as plt
from random import random
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



def addNoise(dataSet, percentage):
	[rows, cols]   = np.shape(dataSet[0])
	pixelsAffected = rows * cols * percentage / 100 
	for im in dataSet:
		indices = []
		while len(indices) < pixelsAffected:
			r = math.ceil(random() * rows)
			c = math.ceil(random() * cols)
			if((r,c) not in indices):
				indices += [(r,c)]

		for (i,j) in indices:
			im[i][j] = np.round(random()) * im[i][j]

	return dataSet
	



def normalizeData(dataset):
	pass


if __name__ == "__main__":
	dataSet          = readData('train-images-idx3-ubyte')
	noise5Data       = addNoise(dataSet, 5)
	noise10Data      = addNoise(noise5Data, 5)
	noise15Data      = addNoise(noise10Data, 5)
	noise20Data      = addNoise(noise15Data, 5)
	noise25Data      = addNoise(noise20Data, 5)
	noise30Data      = addNoise(noise25Data, 5)

	inputComponents  = reduce((lambda x,y : x*y), np.shape(dataSet[0]))
	outputComponents = inputComponents / 2

	network          = FeedForwardNetwork()
	inputLayer       = LinearLayer(inputComponents)
	hiddenLayer      = SigmoidLayer(inputComponents)
	outputLayer      = SigmoidLayer(outputComponents)

	network.addInputModule(inputLayer)
	network.addModule(hiddenLayer)
	network.addOutputModule(outputLayer)

	in_hidden		 = FullConnection(inputLayer, hiddenLayer)
	hidden_out		 = FullConnection(hiddenLayer, outputLayer)

	network.sortModules()





