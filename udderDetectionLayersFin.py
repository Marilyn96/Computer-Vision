import numpy as np
import random
import math


def padding(pad, image, f):
    if pad == 'valid':
        return image
    elif pad == 'same':
        p = (f - 1) / 2
        if p.is_integer():
            p = int(p)
            newImage = np.zeros((len(image) + p * 2, len(image) + p * 2))
            for i in range(len(image)):  # the interior of the image rows
                for j in range(len(image)):  # the interior of the image columns
                    newImage[i + p][j + p] = image[i][j]
        else:
            raise TypeError('The filter size has to be an odd integer value')
    else:  # if pad is 'custom'
        p = f - 1
        checkInt = isinstance(p, int)
        if checkInt:
            p = int(p)
            newImage = np.zeros((len(image) + p * 2, len(image) + p * 2))
            for i in range(len(image)):  # the interior of the image rows
                for j in range(len(image)):  # the interior of the image columns
                    newImage[i + p][j + p] = image[i][j]
        else:
            raise TypeError('The filter size has to be an odd integer value')
    return newImage  # returns the padded image with dimensions (x+p*2,x+p*2) where x is the original dimension


def singleConvolution(kernel_size, f, inputImage, currStride):
    input_shape = inputImage.shape[0]
    outputShape = (input_shape - kernel_size) / currStride + 1
    if outputShape.is_integer():
        outputShape = int(outputShape)
        tempInput = np.zeros((kernel_size, kernel_size))
        currFilter = f
        featureMat = np.zeros((outputShape, outputShape))
        for i in range(outputShape):  # this is the row traversal of the original image
            for g in range(outputShape):  # this is the col traversal of the original image
                for j in range(kernel_size):  # this is the row traversal for the temporary input matrix
                    for k in range(kernel_size):  # this is the col traversal of the temporary input matrix
                        row = i + j
                        col = g + k
                        tempInput[j][k] = inputImage[row][col]

                # compute the convolution of the temporary input and the filter
                tempFeature = 0
                for p in range(kernel_size):
                    t = 0
                    for q in range(kernel_size):
                        t += tempInput[p][q] * currFilter[p][q]
                    tempFeature += t
                featureMat[i][g] = tempFeature
        return featureMat
    else:
        raise TypeError("The stride does not suit the input and filter dimensions")


def rotate180Clockwise(mat):
    size = mat.shape[0]  # N = size , A = mat
    rep = 0
    while rep < 2:
        for i in range(size // 2):
            for j in range(i, size - i - 1):
                temp = mat[i][j]
                mat[i][j] = mat[size - 1 - j][i]
                mat[size - 1 - j][i] = mat[size - 1 - i][size - 1 - j]
                mat[size - 1 - i][size - 1 - j] = mat[j][size - 1 - i]
                mat[j][size - 1 - i] = temp
        rep += 1
    return mat


def fullConv(kernel_size, f, dLdO, stride):
    input_shape = dLdO.shape[0]  # dLdO is a single channel "input image"
    outputShape = (input_shape - 1) * stride + kernel_size  # this is the dL/dX "output feature"
    checkInt = isinstance(outputShape, int)
    if checkInt:
        outputShape = int(outputShape)
        paddedInput = padding('custom', dLdO, kernel_size)
        tempInput = np.zeros((kernel_size, kernel_size))
        currFilter = f
        featureMat = np.zeros((outputShape, outputShape))
        for i in range(outputShape):  # this is the row traversal of the original image
            for g in range(outputShape):  # this is the col traversal of the original image
                for j in range(kernel_size):  # this is the row traversal for the temporary input matrix
                    for k in range(kernel_size):  # this is the col traversal of the temporary input matrix
                        row = i + j
                        col = g + k
                        tempInput[j][k] = paddedInput[row][col]

                # compute the convolution of the temporary input and the filter
                tempFeature = 0
                for p in range(kernel_size):
                    t = 0
                    for q in range(kernel_size):
                        t += tempInput[p][q] * currFilter[p][q]
                    tempFeature += t
                featureMat[i][g] = tempFeature
        return featureMat
    else:
        raise TypeError("The stride does not suit the input and filter dimensions")


class ConvLayer(object):
    def __init__(self, numFilters, filters, kernel_size, stride, bias):
        self.numFilters = numFilters
        self.filters = filters
        self.kernelSize = kernel_size
        self.stride = stride
        self.inputImg = None
        self.paddedImg = None
        self.biases = bias
        self.outputFeatures = None
        self.reluFeatures = None
        self.numInputChannels = None
        self.dLdF = None
        self.dLdX = None

    def convolution(self,  inputImg, numInputChannels, pad):
        # initialize the bias term. All biases are set to zero.
        # Refer to https://stats.stackexchange.com/questions/304287/
        # bias-initialization-in-convolutional-neural-network
        self.inputImg = inputImg
        self.numInputChannels = numInputChannels
        if pad:
            if numInputChannels > 1:
                inputImgChannels = []
                for i in range(numInputChannels):
                    img = padding('same', inputImg[i], self.kernelSize)
                    inputImgChannels.append(img)
                self.paddedImg = np.array(inputImgChannels)
                inputImg = self.paddedImg
            else:
                img = padding('same', inputImg, self.kernelSize)
                self.paddedImg = img
                inputImg = self.paddedImg
        isFilterChannels = False  # This determines whether each filter has more than one channel
        if len(inputImg.shape) == 2:
            input_shape = self.paddedImg.shape[0]
        else:
            input_shape = self.paddedImg.shape[1]
            isFilterChannels = True

        # the convolutional operation
        # checks if the result of (N-F)/s + 1 is an integer, else throws an exception
        outputShape = (input_shape - self.kernelSize) / self.stride + 1
        if outputShape.is_integer():
            outputShape = int(outputShape)
            features = []
            tempInput = np.zeros((self.kernelSize, self.kernelSize))
            if isFilterChannels:
                for f in range(self.numFilters):
                    currFilter = self.filters[f]
                    featureMat = np.zeros((outputShape, outputShape))
                    for i in range(outputShape):  # this is the row traversal of the original image
                        for g in range(outputShape):  # this is the col traversal of the original image
                            intermediateFeature = 0  # an cumulative sum of all temporary features per channel
                            for x in range(numInputChannels):
                                currFilterChannel = currFilter[x]
                                currImageChannel = inputImg[x]  # The current image channel
                                for j in range(self.kernelSize):  # this is the row traversal for the temporary input matrix
                                    for k in range(self.kernelSize):  # this is the col traversal of the temporary input matrix
                                        row = i + j
                                        col = g + k
                                        tempInput[j][k] = currImageChannel[row][col]
                                # compute the convolution of the temporary input and the filter
                                tempFeature = 0
                                for p in range(self.kernelSize):
                                    t = 0
                                    for q in range(self.kernelSize):
                                        t += tempInput[p][q] * currFilterChannel[p][q]
                                    tempFeature += t
                                intermediateFeature += tempFeature
                            intermediateFeature += self.biases[f]
                            featureMat[i][g] = intermediateFeature
                    features.append(featureMat)
                features = np.array(features)
                self.outputFeatures = features
            else:
                for f in range(self.numFilters):
                    if len(self.filters.shape) > 3:
                        currFilter = self.filters[f][0]
                    else:
                        currFilter = self.filters[f]
                    featureMat = np.zeros((outputShape, outputShape))
                    # print('This is filter ' + str(f), currFilter)
                    # print('This is the input image: ', inputImg)
                    for i in range(outputShape):  # this is the row traversal of the original image
                        for g in range(outputShape):  # this is the col traversal of the original image
                            for j in range(self.kernelSize):  # this is the row traversal for the temporary input matrix
                                for k in range(self.kernelSize):  # this is the col traversal of the temporary input matrix
                                    row = i + j
                                    col = g + k
                                    tempInput[j][k] = inputImg[row][col]
                            # compute the convolution of the temporary input and the filter
                            tempFeature = 0
                            # print('This is the temp image:', tempInput)
                            # print('This is the current filter:', currFilter)
                            for p in range(self.kernelSize):
                                t = 0
                                for q in range(self.kernelSize):
                                    t += tempInput[p][q] * currFilter[p][q]
                                tempFeature += t
                            tempFeature += self.biases[f]
                            featureMat[i][g] = tempFeature
                            # print('The matrix element', featureMat[i][g])
                            # print('The current feature matrix:', featureMat)
                    features.append(featureMat)
                    # print('The current feature matrix:', featureMat)
                features = np.array(features)
                self.outputFeatures = features
                # print("The shape of the feature map: ", features.shape[1])
        else:
            raise TypeError("The stride does not suit the input and filter dimensions")

    def activationRelu(self):
        shape = self.outputFeatures.shape
        feats = np.empty(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # features[i][j][k] = np.tanh(features[i][j][k])
                    """
                    if 1 + np.exp(-1 * self.outputFeatures[i][j][k]) == 0:
                        feats[i][j][k] = 0
                    else:
                        feats[i][j][k] = 1 / (1 + np.exp(-1 * self.outputFeatures[i][j][k]))
                    """
                    if self.outputFeatures[i][j][k] < 0:
                        feats[i][j][k] = 0
                    else:
                        feats[i][j][k] = self.outputFeatures[i][j][k]

        self.reluFeatures = feats


class PoolLayer(object):
    def __init__(self):
        self.poolCoordinates = None
        self.poolStride = None
        self.poolKernelSize = None
        self.pooledFeatures = None
        self.reversePooled = None

    def maxPooling(self, features):
        # print("*********** \n The features: \n",features)
        numFeatures = features.shape[0]
        featureSize = features.shape[1]
        final_features = []
        poolCoord = []
        finalPoolCoord = []
        if featureSize % 2 == 0:  # if it is even
            kernel_size = 2
            stride = 2
        else:  # if it is odd
            kernel_size = 3
            stride = 2
        outputShape = int((featureSize - kernel_size) / stride + 1)
        temp_feature = np.zeros((kernel_size, kernel_size))
        for f in range(numFeatures):
            row = 0
            curr_feature = features[f]
            new_feature = np.zeros((outputShape, outputShape))
            for i in range(0, featureSize, stride):  # this is the row traversal of the original feature map
                col = 0
                if row == featureSize - 1:
                    break
                for g in range(0, featureSize, stride):  # this is the col traversal of the original feature map
                    if col == featureSize - 1:
                        break
                    for j in range(0, kernel_size):  # this is the row traversal for the temporary feature map
                        for k in range(0, kernel_size):  # this is the col traversal of the temporary feature map
                            row = i + j
                            col = g + k
                            temp_feature[j][k] = curr_feature[row][col]
                    # compute the max pool of the temporary feature
                    max_ = np.max(temp_feature)
                    result = np.where(temp_feature == max_)
                    argumentRow = result[0][0] + i
                    argumentCol = result[1][0] + g
                    tempPoolCoord = []
                    tempPoolCoord.append(argumentRow)
                    tempPoolCoord.append(argumentCol)
                    poolCoord.append(tempPoolCoord)
                    new_feature[int(i / 2)][int(g / 2)] = max_
            finalPoolCoord.append(poolCoord)
            final_features.append(new_feature)
            poolCoord = []
        final_features = np.array(final_features)

        # print("The pooled feature map: ", final_features)
        self.pooledFeatures = final_features
        self.poolCoordinates = finalPoolCoord
        self.poolStride = stride
        self.poolKernelSize = kernel_size

    def maxPoolingReverse(self, features):
        outputShape = ((features.shape[1] - 1) * self.poolStride) + self.poolKernelSize
        finalFeatures = []
        for i in range(features.shape[0]):  # for each feature map, begin to retrieve original feature map
            currFeature = features[i]
            newFeature = np.zeros((outputShape, outputShape))
            rowO = 0
            colO = 0
            currPosList = self.poolCoordinates[i]
            for j in range(len(currPosList)):  # for each of the coordinates of the positions list
                currCoord = currPosList[j]
                row = currCoord[0]
                col = currCoord[1]
                newFeature[row][col] = currFeature[rowO][colO]
                if colO < features.shape[1] - 1:
                    colO += 1
                elif rowO < features.shape[1] - 1:
                    rowO += 1
                    colO = 0
                else:
                    break
            finalFeatures.append(newFeature)
        self.reversePooled = np.array(finalFeatures)


class FlatLayer(object):
    def __init__(self):
        self.flatFeatures = None
        self.numFeatures = 0
        self.featureSize = 0
        self.deltas = None
        self.unflatGradients = None
        self.numNeurons = 0
        self.neurons = None
        self.bias = 0

    def flatten(self, unflattenedFeatures):
        flat = []
        self.numFeatures = unflattenedFeatures.shape[0]
        self.featureSize = unflattenedFeatures.shape[1]

        for f in range(self.numFeatures):
            curr_feature = unflattenedFeatures[f]
            for i in range(self.featureSize):
                for j in range(self.featureSize):
                    flat.append(curr_feature[i][j])
        flat2 = np.zeros((len(flat)+1, 1))
        for i in range(len(flat)):
            flat2[i][0] = flat[i]
        flat2[len(flat)] = 1  # Flat layer bias
        self.bias = 1

        self.numNeurons = flat2.size
        self.neurons = flat2
        self.flatFeatures = flat2

    def unflattenGradients(self):
        unflatGradients = []
        i = 0
        for j in range(self.numFeatures):
            currFeature = np.zeros((self.featureSize, self.featureSize))
            for k in range(self.featureSize):
                for l in range(self.featureSize):
                    currFeature[k][l] = self.deltas[i]
                    i += 1
            unflatGradients.append(currFeature)
        self.unflatGradients = np.array(unflatGradients)


def annActivation(input_):
    # return np.tanh(input_)
    """
    if 1 + np.exp(-1 * input_) == 0:
        return 0
    else:
        return 1 / (1 + np.exp(-1 * input_))
    """
    if input_ < 0:
        return 0
    else:
        return input_


class HiddenLayer(object):
    def __init__(self, numNeurons, numInputs):
        self.numNeurons = numNeurons+1
        self.numInputs = numInputs+1
        self.inputs = None  # this is the previous layer's neurons array
        self.weights = None
        self.neurons = None
        self.deltas = None
        self.bias = 0

    def hiddenWeightInit(self, numHiddenLayer):
        # numWeights = numInputs * numInputs
        outputs = np.loadtxt(open("TrainedUdderDetectionFinal/DenseLayer" + str(numHiddenLayer) + ".csv", "rb"),delimiter=",")
        biases = np.loadtxt(open("TrainedUdderDetectionFinal/Dense" + str(numHiddenLayer) + "Bias.csv", "rb"),delimiter=",")
        weights = np.zeros((self.numNeurons-1, self.numInputs))  # the -1 indicates the weights from the bias to the next bias is excluded
        for i in range(self.numNeurons-1):
            for j in range(self.numInputs-1):
                weights[i][j] = outputs[i][j]
            weights[i][2048] = biases[i]
        outputs = np.zeros((self.numNeurons, 1))
        outputs[self.numNeurons-1] = 1
        self.bias = 1

        self.neurons = outputs
        self.weights = weights

    def calculateNeurons(self, inputs):
        for i in range(self.numNeurons-1):
            summing = 0
            for j in range(self.numInputs):
                summing += self.weights[i][j] * inputs[j]
            self.neurons[i][0] = annActivation(summing)


class OutputLayer(object):
    def __init__(self, numNeurons, numInputs):
        self.numNeurons = numNeurons
        self.numInputs = numInputs+1
        self.weights = None
        self.neurons = None
        self.deltas = None

    def outputWeightsInit(self, numHiddenLayer):
        outputs = np.loadtxt(open("TrainedUdderDetectionFinal/DenseLayer" + str(numHiddenLayer) + ".csv", "rb"),delimiter=",")
        biases = np.loadtxt(open("TrainedUdderDetectionFinal/Dense" + str(numHiddenLayer) + "Bias.csv", "rb"),delimiter=",")
        weights = np.zeros((self.numNeurons, self.numInputs))
        for i in range(self.numNeurons):
            for j in range(self.numInputs-1):
                weights[i][j] = outputs[i][j]
            weights[i][50] = biases[i]
        outputs = np.zeros((self.numNeurons, 1))

        self.neurons = outputs
        self.weights = weights

    def calculateNeurons(self, inputs):
        for i in range(self.numNeurons):
            summing = 0
            for j in range(self.numInputs):
                summing += self.weights[i][j] * inputs[j]
            self.neurons[i] = summing

