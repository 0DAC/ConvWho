""" Provides the Convolutional Neural Network classifiers for gender
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as grapher
import argparse


class GenderClassifier:
    """ Network structure for classifying male/female gender from an input photo """

    def __init__(self, dimension_x, dimension_y):
        self.labels = ['Male', 'Female']

        # Configuring the CNN model ------------------------------------------------------------

        # TODO: Optimize network structure for CNN image classification
        self.model = keras.Sequential([
            # Input layer converts image to a 2D array
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            # Output later outputs two probabilities for male/female
            keras.layers.Dense(10, activation='softmax')
        ])

        # Settings:
        # 1) Optimizer: how the model updates based on input data and loss
        # 2) Loss func: measures model accuracy during training, goal is to minimize this output
        # 3) Metrics: reward factor
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, training_images, training_labels, epochs=1):
        """
        Trains the neural network based on paired training data/labels

        :param training_images: set of images (must be same length as training_labels)
        :param training_labels: set of labels (must be same length as training_images)
        :param epochs: number of training iterations
        :return: None
        """

        if not len(training_labels) == len(training_images):
            print(
                "Error in GenderClassifier Neural Network Training: training image and label set length do not match")
            return

        self.model.fit(training_images, training_labels, epochs=epochs)

    def test(self, testing_images, testing_labels):
        """
        Evaluates accuracy of current neural network model

        :param testing_images: set of unrevealed images (must be same length as testing_labels)
        :param testing_labels: set of image labels      (must be same length as testing_images)
        :return: testing_loss (type=float), testing_accuracy (type=float)
        """

        if not len(testing_labels) == len(testing_images):
            print(
                "Error in GenderClassifier Neural Network Training: testing image and label set length do not match")
            return None, None

        return self.model.evaluate(testing_images, testing_labels)


# ONLY IF GenderClassifier works
class EthnicClassifier:
    """ Network structure for classifying ethnic background from an input photo """

class ExampleClassifier:
    """ Tutorial classifier from https://www.tensorflow.org/tutorials/keras/basic_classification """

    def __init__(self, dimension_x, dimension_y):
        """ Set up an untrained neural network structure from hard-coded structure """
        self.labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Configuring the CNN model ------------------------------------------------------------

        # Input layer (type=Flatten) converts image to a 2D array
        # Output later (type=Dense, activation=softmax) outputs two probabilities for male/female
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Settings:
        # 1) Optimizer: how the model updates based on input data and loss
        # 2) Loss func: measures model accuracy during training, goal is to minimize this output
        # 3) Metrics: reward factor
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print("ExampleClassifier Neural Network Setup Complete------------------------------------------------------------")
        self.model.summary()

    def load_saved_model(cls, path):
        # TODO: implement how to load model
        """
        Alternate constructor to load a pre-trained model

        :param path: relative path to model file
        :return: None
        """
        pass

    def save_model(self, filepath):
        # TODO: implement how to save model
        """
        Saves trained NN structure

        :param filepath: relative path to NN model file
        :return: None
        """
        pass

    def train(self, training_images, training_labels, epochs=1):
        """
        Trains the neural network based on paired training data/labels

        :param training_images: set of images (must be same length as training_labels)
        :param training_labels: set of labels (must be same length as training_images)
        :param epochs: number of training iterations
        :return: None
        """

        if not len(training_labels) == len(training_images):
            print("Error in ExampleClassifier Neural Network Training: training image and label set length do not match")
            return

        self.model.fit(training_images, training_labels, epochs=epochs)

    def test(self, testing_images, testing_labels):
        """
        Evaluates accuracy of current neural network model

        :param testing_images: set of unrevealed images (must be same length as testing_labels)
        :param testing_labels: set of image labels      (must be same length as testing_images)
        :return: testing_loss (type=float), testing_accuracy (type=float)
        """

        if not len(testing_labels) == len(testing_images):
            print(
                "Error in ExampleClassifier Neural Network Training: testing image and label set length do not match")
            return None, None

        return self.model.evaluate(testing_images, testing_labels)

    def predict(self, input_images):
        """
        Assigns an (ideally untested) image a label

        :param input_image: list of preprocessed (properly sized and flattened) images
        :return: array of possibilites. Find the largest value with np.argmax(predictions[i])
        """

        return self.model.predict(input_images)


def debug():
    print(">--------------<\nDebugging cnn.py\n>--------------<")
    print("Currently using Tensorflow", tf.__version__)
    batch = 32
    training_epochs = 5
    verbose = True
    training = True
    testing = True
    predicting = True
    running_tutorial = True
    running_gender = False

    if running_tutorial:
        if verbose:
            print("Running Tutorial Classifier----------------------------------------------------------------------------")
        # the example classifier utilizes the Keras fasion dataset
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        classifier = ExampleClassifier(train_images.shape[1], train_images.shape[2])

        if verbose:
            print("Keras MNIST Dataset:")
            print("%d training images" %
                  train_images.shape[0])
            print("%d testing images\nImage dimensions: %d px by %d px" %
                  test_images.shape)

        # Displaying images via matplotlib
        # grapher.figure()
        # grapher.imshow(train_images[0])
        # grapher.colorbar()
        # grapher.grid(False)
        # grapher.show()

        # normalize values to [0, 1]
        train_images = train_images/255
        test_images = test_images/255

        # show the first 25 training images
        # grapher.figure(figsize=(10, 10))
        # for photo in range(25):
        #     grapher.subplot(5, 5, photo+1)
        #     grapher.xticks([])
        #     grapher.yticks([])
        #     grapher.grid(False)
        #     grapher.imshow(train_images[photo], cmap=grapher.cm.binary)
        #     grapher.xlabel(classifier.labels[train_labels[photo]])
        # grapher.show()

        # Training Day

        if training:
            if verbose:
                print("Training Neural Network Model")
            classifier.train(train_images, train_labels, epochs=training_epochs)

        # Evaluate accuracy
        if testing:
            if verbose:
                print("Testing Neural Network Model")
            test_loss, test_acc = classifier.test(test_images, test_labels)
            print("Test accuracy: ", test_acc)

        if predicting:
            if verbose:
                print("Predicting Result from Neural Network Model")

            # NOTE: it is not good practice to predict a test-image
            predictions = classifier.predict(test_images)

            # for i, item_prediction in enumerate(predictions):
            #    print("Prediction for Image", i, " is ", classifier.labels[np.argmax(item_prediction)])
            #    print("The correct classification for Image", i, " is ", classifier.labels[test_labels[i]])



    elif running_gender:
        pass
if __name__ == '__main__':
    debug()

