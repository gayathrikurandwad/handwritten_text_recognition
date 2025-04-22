"""
For Testing purposes
    Take image from user, crop the background and transform perspective
    from the perspective detect the word and return the array of word's
    bounding boxes
"""

import page
import words
from PIL import Image
import cv2

def segment_para():
    # User input page image
    image = cv2.cvtColor(cv2.imread("static/images/3.png"), cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(1950,1000))
    # Crop image and get bounding boxes
    crop = page.detection(image)
    boxes = words.detection(crop)
    lines = words.sort_words(boxes)

    # Saving the bounded words from the page image in sorted way
    i = 0
    for line in lines:
        text = crop.copy()
        for (x1, y1, x2, y2) in line:
            # roi = text[y1:y2, x1:x2]
            save = Image.fromarray(text[y1:y2, x1:x2])

            # print(i)
            save.save("segmented/segment" + str(i) + ".png")
            i += 1



