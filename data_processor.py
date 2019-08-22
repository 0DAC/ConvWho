"""

Defines the Human class with which we will work later

Provides general information about all the humans in a frame such as:
    raw_data(img)
    rect_area(tuple)

"""
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2


class Human:
    """""
    This is the class that will store information for the identified humans.
        Raw_data : img
        gender_tag : Male/Female
        rect_area : a tuple of the coordinates of the rectangle
    """""

    def __init__(self):
        self.raw_data = None
        self.gender_tag = None
        self.rect_area = []


class DataProcessor(object):
    """
    This class processes an image given by detecting and extracting the humans in a picture
    """

    def __init__(self):
        pass

    def detect_human(self, image_path="", img=None):
        """
        Using hog and the .detectMultiScale function, this function gives us the surface area in which a human is situated
        :param image_path:
        :return: a list of rectangular coordinates with the format: [x_upper_left, y_upper_left, x_low_right, y_low_right]
        """

        hog = cv2.HOGDescriptor()

        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        image = img
        image = imutils.resize(image, width=min(400, image.shape[1]))
        original = image.copy()

        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # for (x, y, w, h) in rects:
        #     cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # for (xA, yA, xB, yB) in pick:
        #     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        filename = image_path[image_path.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
        print(rects)
        # cv2.imshow("After recognition", image)
        # cv2.waitKey(0)

        return rects

    def crop_human(self, rectangular_areas, original_path="", img=None):
        """
        The function also saves all the humans in the directory humans
        :param rectangular_areas: The list returned by the function detect_human
        :param original_path: The original image
        :param img: cv2 object
        :return: Img
        """

        "Resizing the image to the format used in the detect_human function"
        image = img

        cropped_img = []

        "Cropping the image"
        for (x1, y1, x2, y2) in rectangular_areas:
            image = imutils.resize(image, width=min(400, image.shape[1]))
            cropped_img.append(image[y1:y2, x1:x2])
            # cropped_img.save

        cv2.imshow("Fuck" ,cropped_img[0])
        cv2.waitKey(0)

        return cropped_img


if __name__ == "__main__":
    "Initializing the classes"
    test = DataProcessor()
    person = Human

    person.rect_area = test.detect_human(image_path='sample_images/mafia.jpeg')

    test.crop_human(person.rect_area, 'sample_images/mafia.jpeg')
    # person.raw_data = test.crop_human(person.rect_area, 'sample_images/mafia.jpeg')
