import udderDetectionModelFin
import numpy as np

########################################################################################################################
# Create the CNN Model
########################################################################################################################
def defineModel():
    numFilters = [8, 16, 32]
    model = udderDetectionModelFin.CNN(numFilters, [3, 3, 3], [1, 1, 1])
    model.add("ConvLayer", inputImgShape=[64, 64])  # ConvLayer1
    model.add("Pooling")  # PoolLayer1
    model.add("ConvLayer", inputImgShape=[8, 32, 32])  # ConvLayer2
    model.add("Pooling")  # PoolLayer2
    model.add("ConvLayer", inputImgShape=[16, 16, 16])  # ConvLayer 3
    model.add("Pooling")  # PoolLayer 3
    model.add("Flatten")  # Flat layer
    model.add("Hidden", numNeurons=50, numInputNeurons=2048)  # HiddenLayer 1
    # model.add("Hidden", numNeurons=1000, numInputNeurons=1000)  # HiddenLayer 2
    model.add("Output", numNeurons=2, numInputNeurons=50)  # OutputLayer
    return model


def predict(image):
    m = defineModel()
    prediction = m.predict(image)
    return prediction
