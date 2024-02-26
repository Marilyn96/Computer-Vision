# The full main system code
import SerialFin
import SelectiveSearchFin
from PIL import Image, ImageEnhance
import udderDetectionMainFin as udder
import teatDetectionMainFin as teat
import teatClassificationMainFin as classification
import os
import cv2
import PreprocessingFin
import numpy as np
import Database

# Run the main code for the camera to capture an image
SerialFin.main()  # Outputs a 640x480 image called UdderImage.jpg
image = Image.open('UdderImage.jpg')
enhancer_object = ImageEnhance.Brightness(image)
out = enhancer_object.enhance(5)
out.save('UdderImage.jpg')
address = 'UdderImage.jpg'
response = Database.sendImage(address, [0, 0, 0, 0, 0])


# The image does not need to be cropped. Resize image to size = (64, 64)
image = Image.open('udderImage.jpg')
image = image.resize((64, 64))
image = PreprocessingFin.main(np.array(image), 130, 160)
#image = Image.fromarray(image)
#image.save('thumbnail.jpg')
image = image/255.0
# Send image to udder detection code
chosenClass = udder.predict(image)
print(chosenClass)
print('Udder detection done')

image = Image.open('UdderImage.jpg')
if not os.path.exists('thumbnails'):
    os.makedirs('thumbnails')
# Apply selective search to the image
boxes = SelectiveSearchFin.createSearchBoxed("C:/Users/test/Documents/Academics/2021/EPR402/Final_Code/UdderImage.jpg")
print(boxes[0].shape)
images = boxes
for i in range(len(images)):
    images[i] = cv2.cvtColor(boxes[i], cv2.COLOR_BGR2RGB)
    cv2.imwrite('thumbnails/'+str(i)+'.jpg', images[i])
print("Selective search done")

if not os.path.exists('thumbnails2'):
    os.makedirs('thumbnails2')

# Pad the image with black background
images = []
size = (50, 50)
for i in range(80):
    image = Image.open('thumbnails/' + str(i) + '.jpg')
    image = np.array(image)
    image = PreprocessingFin.main(image, 130, 160)
    image = Image.fromarray(image)
    image = PreprocessingFin.expand2square(image, 'white').resize(size)
    images.append(image)
    image.save('thumbnails2/' + str(i) + '.jpg')
print('Image padding done')

# Put images through the udder detection code
classes = []
teats = []
if not os.path.exists('segteats'):
    os.makedirs('segteats')
j = 0
for i in range(80):
    image = Image.open('thumbnails2/' + str(i) + '.jpg')
    image = np.array(image)
    image2 = image/255.0
    chosenClass = teat.predict(image2)
    classes.append(chosenClass)
    print(chosenClass)
    if chosenClass == 1:
        image = Image.fromarray(image)
        image.save('segteats/' + str(j) + '.jpg')
        j += 1
    if j == 4:
        break

print('Teat detection done')


scores = []
for i in range(4):
    image = Image.open('segteats/'+ str(i) + '.jpg')
    image = image.resize((100, 100))
    image = np.array(image)
    chosenClass = classification.predict(image)
    scores.append(chosenClass+1)
average = sum(scores)/4
address = 'UdderImage.jpg'
response = Database.sendImage(address, [average, scores[0], scores[1], scores[2], scores[3]])

