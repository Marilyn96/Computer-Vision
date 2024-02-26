# This is the CNN model framework which allows the CNN to be configured much easier
import udderDetectionLayersFin
import numpy as np
import math
from PIL import Image, ImageOps
from sklearn.utils import shuffle
from statistics import mean
import random
import matplotlib.pyplot as plt


def softMax(outputPrediction):
    denominator = np.sum([np.exp(o) for o in outputPrediction])
    return [np.exp(o)/denominator for o in outputPrediction]


def crossEntropyLoss(actualValue, outputPrediction):
    """
    index = actualValue.index(1)
    denominator = np.sum([np.exp(o) for o in outputPrediction])
    loss = -np.log(outputPrediction[index]/denominator)
    return loss
    """
    # denominator = np.sum([np.exp(o) for o in outputPrediction])
    # final = np.exp(outputPrediction[index]) / denominator
    loss = -np.sum(actualValue * np.log(outputPrediction))
    return loss


def annActivationDeriv(input_):
    denom = 1 + np.exp(-1 * input_)
    fx = 1 / denom
    """
    if input_ <= 0:
        return 0
    else:
        return 1

    # return 1 - pow(np.tanh(input_), 2)
    """
    return fx * (1 - fx)


class CNN(object):
    def __init__(self, numFilters, filterKernels, strides):
        self.architecture = dict()
        self.numConvLayers = 0
        self.numPoolingLayers = 0
        self.numFilters = numFilters
        self.filterKernels = filterKernels
        self.strides = strides
        self.numHiddenLayers = 0
        self.neuronsPerHidden = 0
        self.filters = []
        self.features = []
        self.weights = []
        self.neurons = []
        self.poolLocations = []
        self.lossPerBatch = 0
        self.testLabels = None
        self.testImages = None
        self.trainLabels = None
        self.trainImages = None
        self.gradients = None
        self.numOutputNeurons = 0
        self.accuracy = 0

    # initializes filters. It creates a 2 dimensional array of filters for the current conv layer(an array of matrices)

    def add(self, layerName, inputImgShape=None, numNeurons=None, numInputNeurons=None):
        if layerName == 'ConvLayer':
            self.numConvLayers += 1
            currLayerIndex = self.numConvLayers
            filters = []
            if len(inputImgShape) == 2:
                for i in range(self.numFilters[currLayerIndex-1]):
                    filters.append(np.loadtxt(open("TrainedUdderDetectionFinal/Conv" + str(self.numConvLayers) + "Kernel" + str(i) + ".csv", "rb"), delimiter=","))
                biases = np.loadtxt(open("TrainedUdderDetectionFinal/Conv" + str(self.numConvLayers) + "Bias.csv", "rb"), delimiter=",")
            else:
                for i in range(self.numFilters[currLayerIndex - 1]):
                    filterChannels = []
                    for j in range(self.numFilters[currLayerIndex - 2]):
                        filterChannels.append(np.loadtxt(open("TrainedUdderDetectionFinal/Conv" + str(self.numConvLayers) + "Kernel" + str(i) + "Channel"+ str(j)+".csv","rb"), delimiter=","))
                    filters.append(np.array(filterChannels))
                biases = np.loadtxt(open("TrainedUdderDetectionFinal/Conv" + str(self.numConvLayers) + "Bias.csv", "rb"),delimiter=",")

            ConvLayer = udderDetectionLayersFin.ConvLayer(self.numFilters[currLayerIndex-1], np.array(filters), self.filterKernels[currLayerIndex-1], self.strides[currLayerIndex-1], biases)
            # ConvLayer.convolution()
            # ConvLayer.activationRelu()
            # self.features.append(ConvLayer.outputFeatures)
            # self.features.append(ConvLayer.reluFeatures)
            self.architecture["ConvLayer{0}".format(self.numConvLayers)] = ConvLayer  # Adds this layer to dictionary
        elif layerName == 'Pooling':
            self.numPoolingLayers += 1
            PoolLayer = udderDetectionLayersFin.PoolLayer()
            # PoolLayer.maxPooling()
            # self.features.append(PoolLayer.pooledFeatures)
            # self.poolLocations.append(PoolLayer.poolCoordinates)
            self.architecture["PoolLayer{0}".format(self.numPoolingLayers)] = PoolLayer  # Adds this layer to dictionary
        elif layerName == 'Flatten':
            FlatLayer = udderDetectionLayersFin.FlatLayer()
            self.architecture["FlatLayer"] = FlatLayer
        elif layerName == 'Hidden':
            self.numHiddenLayers += 1
            HiddenLayer = udderDetectionLayersFin.HiddenLayer(numNeurons, numInputNeurons)
            HiddenLayer.hiddenWeightInit(self.numHiddenLayers)
            self.architecture["HiddenLayer{0}".format(self.numHiddenLayers)] = HiddenLayer
        else:
            OutputLayer = udderDetectionLayersFin.OutputLayer(numNeurons, numInputNeurons)
            self.numOutputNeurons = numNeurons
            OutputLayer.outputWeightsInit(self.numHiddenLayers+1)
            self.architecture["OutputLayer"] = OutputLayer

    def predict(self, image):
        numInputChannels = 1
        ConvLayer1 = self.architecture["ConvLayer1"]
        ConvLayer1.convolution(image, numInputChannels, True)
        ConvLayer1.activationRelu()
        features = ConvLayer1.reluFeatures
        PoolLayer1 = self.architecture["PoolLayer1"]
        PoolLayer1.maxPooling(features)
        features = PoolLayer1.pooledFeatures

        numInputChannels = features.shape[0]
        ConvLayer2 = self.architecture["ConvLayer2"]
        ConvLayer2.convolution(features, numInputChannels, True)
        ConvLayer2.activationRelu()
        features = ConvLayer2.reluFeatures
        PoolLayer2 = self.architecture["PoolLayer2"]
        PoolLayer2.maxPooling(features)
        features = PoolLayer2.pooledFeatures

        numInputChannels = features.shape[0]
        ConvLayer3 = self.architecture["ConvLayer3"]
        ConvLayer3.convolution(features, numInputChannels, True)
        ConvLayer3.activationRelu()
        features = ConvLayer3.reluFeatures
        PoolLayer3 = self.architecture["PoolLayer3"]
        PoolLayer3.maxPooling(features)
        features = PoolLayer3.pooledFeatures

        FlattenLayer = self.architecture["FlatLayer"]
        FlattenLayer.flatten(features)
        features = FlattenLayer.flatFeatures

        HiddenLayer1 = self.architecture["HiddenLayer1"]
        HiddenLayer1.calculateNeurons(features)
        features = HiddenLayer1.neurons

        # HiddenLayer2 = self.architecture["HiddenLayer2"]
        # HiddenLayer2.calculateNeurons(features)
        # features = HiddenLayer2.neurons

        OutputLayer = self.architecture["OutputLayer"]
        OutputLayer.calculateNeurons(features)
        features = OutputLayer.neurons

        # Creating horizontal list for softmax function
        horizontalFeatures = []
        softFeatures = softMax(features)
        OutputLayer.neurons = np.array(softFeatures)

        if OutputLayer.neurons[0] > OutputLayer.neurons[1]:
            index = 0
        elif OutputLayer.neurons[0] < OutputLayer.neurons[1]:
            index = 1
        else:
            index = 100

        return index




