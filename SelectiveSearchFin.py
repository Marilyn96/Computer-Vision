import selective_search
import numpy as np
from PIL import Image


def createSearchBoxed(address):
    # Load image
    image = Image.open(address)
    image = np.array(image)

    # Propose boxes using selective search
    boxes = selective_search.selective_search(image, mode='fast')

    boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=80)

    images = []
    for x1, y1, x2, y2 in boxes_filter:
        images.append(image[y1:y2, x1:x2])
    return images
