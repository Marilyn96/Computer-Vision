import teatClassificationModelFin
import numpy as np

########################################################################################################################
# Create the CNN Model
########################################################################################################################
def defineModel():
    numFilters = [8, 16, 32, 64]
    model = teatClassificationModelFin.CNN(numFilters, [3, 3, 3, 3], [1, 1, 1, 1])
    model.add("ConvLayer", inputImgShape=[100, 100])  # ConvLayer1
    model.add("Pooling")  # PoolLayer1
    model.add("ConvLayer", inputImgShape=[8, 50, 50])  # ConvLayer2
    model.add("Pooling")  # PoolLayer2
    model.add("ConvLayer", inputImgShape=[16, 25, 25])  # ConvLayer 3
    model.add("Pooling")  # PoolLayer 3
    model.add("ConvLayer", inputImgShape=[64, 12, 12])  # ConvLayer 3
    model.add("Pooling")  # PoolLayer 3
    model.add("Flatten")  # Flat layer
    model.add("Hidden", numNeurons=500, numInputNeurons=2304)  # HiddenLayer 1
    model.add("Hidden", numNeurons=500, numInputNeurons=500)  # HiddenLayer 2
    model.add("Output", numNeurons=4, numInputNeurons=500)  # OutputLayer
    return model


def predict(image):
    m = defineModel()
    prediction = m.predict(image)
    return prediction
