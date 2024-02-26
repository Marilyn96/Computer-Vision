import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageStat, Image
import math
import os


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(mode=pil_img.mode, size=(width, width), color=background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def convertToGrayscale(img, shape):
    # Convert from BGR to grayscale using weighted method
    blue_channel = img[:,:,0]
    red_channel = img[:,:,2]
    green_channel = img[:,:,1]
    gray = np.empty(shape[0:2], dtype='uint8')

    # Convert image to black and white
    for i in range(shape[0]):
        for j in range(shape[1]):
            gray[i][j] = 0.299*red_channel[i][j] + 0.587*green_channel[i][j] + 0.114*blue_channel[i][j]

    # #plt.imshow(gray, cmap='gray')
    # #plt.show()
    return gray  # Image.fromarray(gray.astype(np.uint8), mode='L')


# Determines the RMS image brightness value
def obtainContrast(grayscale, shape):
    minimum = 0
    maximum = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if grayscale[i][j] > maximum:
                maximum = grayscale[i][j]
            elif grayscale[i][j] < minimum:
                minimum = grayscale[i][j]
    contrast = maximum - minimum
    return contrast


def avgBrightness(image, shape):
    # Calculate average intensity/brightness
    b = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            b += image[i][j]
    return b / (shape[0]*shape[1])  # average brightness


def adjustBrightness(image, brightness, shape):
    newImage = np.empty(shape, dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newImage[i][j] = truncate(image[i][j] + brightness)
    # #plt.imshow(newImage, cmap='gray')
    # #plt.show()
    return newImage


def adjustContrast(image, contrast):
    # Contrast is a value betwen -255 to 255
    # Contrast factor
    factor = (259*(contrast+255))/(255*(259-contrast))

    # Contrast adjustment
    newImage = np.empty(image.shape, dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newImage[i][j] = truncate(factor * (image[i][j] - 128) + 128)

    # #plt.imshow(newImage, cmap='gray')
    # #plt.show()
    return newImage


def truncate(value):
    if value < 0:
        return 0
    elif value > 255:
        return 255
    else:
        return value


def main(img, b, c):
    # Obtain image
    #img = cv2.imread("C:/Users/test/Documents/Academics/2021/EPR402/Final_Code/UdderImage.jpg", cv2.IMREAD_UNCHANGED)
    # Convert the image to grayscale
    gray = convertToGrayscale(img, img.shape)

    # Determine the brightness level
    level = avgBrightness(gray, gray.shape)

    # The target brightness is 50, so determine the offset applied to reach this target
    offset = b - level

    # Brighten the image with the specified birghtness level
    brightened = adjustBrightness(gray, offset, gray.shape)
    level = avgBrightness(brightened, brightened.shape)
    # cv2.imwrite("C:/Users/test/Documents/Academics/2021/EPR402/Final_Code/brightened.jpg", brightened)

    ####################################################################################################################
    # Determine contrast
    contrast = obtainContrast(brightened, brightened.shape)

    # Determine offset for contrasting. The target offset is 100
    offset = c - contrast

    # Contrast the image with the specified offset
    contrasted = adjustContrast(brightened, offset)
    contrast = obtainContrast(contrasted, contrasted.shape)
    return contrasted
    #cv2.imwrite("C:/Users/test/Documents/Academics/2021/EPR402/Final_Code/contrasted.jpg", contrasted)



