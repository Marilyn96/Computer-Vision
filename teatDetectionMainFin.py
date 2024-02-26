import teatDetectionModelFin
import numpy as np

########################################################################################################################
# Create the CNN Model
########################################################################################################################
def defineModel():
    numFilters = [8, 16, 32]
    model = teatDetectionModelFin.CNN(numFilters, [3, 3, 3], [1, 1, 1])
    model.add("ConvLayer", inputImgShape=[50, 50])  # ConvLayer1
    model.add("Pooling")  # PoolLayer1
    model.add("ConvLayer", inputImgShape=[8, 25, 25])  # ConvLayer2
    model.add("Pooling")  # PoolLayer2
    model.add("ConvLayer", inputImgShape=[16, 12, 12])  # ConvLayer 3
    model.add("Pooling")  # PoolLayer 3
    model.add("Flatten")  # Flat layer
    model.add("Hidden", numNeurons=60, numInputNeurons=512)  # HiddenLayer 1
    model.add("Hidden", numNeurons=60, numInputNeurons=60)  # HiddenLayer 2
    model.add("Output", numNeurons=2, numInputNeurons=60)  # OutputLayer
    return model


def predict(image):
    m = defineModel()
    prediction = m.predict(image)
    return prediction
