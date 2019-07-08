""" Provides the Convolutional Neural Network classifiers for gender"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as grapher
import argparse

tf.logging.set_verbosity(tf.logging.INFO)

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

class CNN:

    def __init__(self, tf_session, dimension_x, dimension_y, file_loc):
        """
        Initializes basic variables

        :param dimension_x: x size of input monochromatic images
        :param dimension_y: y size of input monochromatic images
        :param file_loc: folder to store checkpoints and saved model
        """
        print("--------------------------------------------------------------")
        print("Initializing basic CNN parameters...")
        self.session = tf_session
        self.input_x = dimension_x
        self.input_y = dimension_y
        self.dir = file_loc
        self.debug = False
        print("Successfully initialized CNN...")
        print("--------------------------------------------------------------")

    def init_network(self, features, labels, mode):
        """
                Creates the CNN network structure and stores it in self.model
                :param features:
                :param mode: either tf.estimator.ModeKeys.TRAIN / EVAL
                :param dimension_x: x dimension of input images
                :param dimension_y: y dimension of input images
                :return: None
                """
        self.features = features
        self.labels = labels

        # Configuring the CNN model ------------------------------------------------------------
        self.special_print("--------------------------------------------------------------")
        self.special_print("Synthesizing CNN Network Structure...")
        # TODO: Optimize network structure for CNN image classification
        self.special_print("\tCreating input layer for images of size %dx%d..." % (self.input_x, self.input_y))
        # Input Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # tensor input: [batch_size (-1 means dynamically compute),
        #                 dim_x,
        #                 dim_y,
        #                 color_channels
        #               ]
        self.input_layer = tf.reshape(features["x"], [-1, self.input_x, self.input_y, 1])

        self.special_print("\tCreating Conv Layer 1...")
        # Convolutional Layer #1
        # Apply 32 5x5 filters
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)

        self.special_print("\tCreating Conv Layer 2...")
        # Convolutional Layer #2 and Pooling Layer #2
        self.conv2 = tf.layers.conv2d(
            inputs=self.pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)

        self.special_print("\tCreating Dense Hidden Layer...")
        # Dense Layer
        self.pool2_flat = tf.reshape(self.pool2, [-1, 7 * 7 * 64])
        self.dense = tf.layers.dense(inputs=self.pool2_flat, units=1024, activation=tf.nn.relu)
        self.dropout = tf.layers.dropout(
            inputs=self.dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        self.special_print("\tCreating Logits Layer...")
        # Logits Layer
        self.logits = tf.layers.dense(inputs=self.dropout, units=10)

        self.predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=self.logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
        }

        # Using the model to Predict ------------------------------------------------------------

        if mode == tf.estimator.ModeKeys.PREDICT:
            self.special_print(">>> Creating Prediction Model...")
            self.model = tf.estimator.EstimatorSpec(mode=mode, predictions=self.predictions)
            return self.model

        # Using the model to TRAIN/EVAL ------------------------------------------------------------

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.special_print(">>> Creating Training Model...")
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            self.model = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            return self.model


        # Add evaluation metrics (for EVAL mode)
        self.special_print(">>> Creating Evaluation Model...")
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=self.labels, predictions=self.predictions["classes"])
        }
        self.model = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return self.model

    def init_from_model(self):
        pass

    def init_classifier(self):
        """
        Creates tf Estimator for high-level training, evaluation, and prediction

        :return: None
        """
        self.special_print("--------------------------------------------------------------")
        self.special_print("Creating Classifier...")
        self.classifier = tf.estimator.Estimator(
            model_fn=self.init_network, # specify model function for Train, Eval, Pred
            model_dir=self.dir # specify where to store checkpoints and models
        )
        self.special_print("Successfully created Classifier...")
        self.special_print("--------------------------------------------------------------")

    def init_progress_tracker(self, track_freq):
        """
        Tracks CNN training progress (which commonly take a while)

        :param track_freq: how many training steps per update cycle
        :return: None
        """
        self.special_print("--------------------------------------------------------------")
        self.special_print("Creating Progress Tracker...")

        # Map custom label (printed in log) to a specific tensor variable name
        self.tensors_to_track = {"probabilities" : "softmax_tensor"}
        self.progress_logger = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_track,
            every_n_iter=track_freq
        )

        self.special_print("Successfully Created Progress Tracker...")
        self.special_print("--------------------------------------------------------------")

    def train(self, training_set, batch_size, training_steps, epochs, log_output=False):
        """
        Function to train the CNN

        :param training_set: tuple of training data and training labels
        :param batch_size: training data size
        :param training_steps: number of training "sections"
        :param epochs: number of training episodes
        :return: None
        """

        self.special_print("--------------------------------------------------------------")
        self.special_print("Initiating Training Process")
        training_data, training_labels = training_set

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':training_data},
            y=training_labels,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=True
        )

        self.classifier.train(
            input_fn=input_fn,
            steps=training_steps,
            hooks=([self.progress_logger] if log_output else None)
        )
        self.special_print("Completed Training")
        self.special_print("--------------------------------------------------------------")

    def evaluate(self, eval_set, epochs):

        """
        Evaluates the current CNN model

        :param eval_set: tuple of ( eval_data, eval_labels )
        :param epochs: number of evaluation episodes
        :return: training results
        """
        eval_data, eval_labels = eval_set
        self.special_print("--------------------------------------------------------------")
        self.special_print("Initiating CNN Model Evaluation")

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=epochs,
            shuffle=False)

        self.special_print("Finished Model Evaluation, returning results")
        self.special_print("--------------------------------------------------------------")

        return self.classifier.evaluate(input_fn=input_fn)

    def set_debug_mode(self, status):
        self.debug = (status==True)
        print("--------------------------------------------------------------")
        print("Debug Status: %s" % (str(self.debug)))
        print("--------------------------------------------------------------")

    def special_print(self, str):
        if self.debug:
            print(str)

def load_local_data(dir):
    """
    [Helper function] Parses data from local source

    :param dir: path to data directory
    :return: ( (train_data, train_labels), (eval_data, eval_labels) )
    """

    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    # compress data to float32
    train_data = train_data/np.float32(255)
    eval_data = eval_data/np.float32(255)

    # compress labels to int32
    train_labels = train_labels.astype(np.int32)
    eval_labels = eval_labels.astype(np.int32)


    return ((train_data, train_labels),
            (eval_data, eval_labels))


if __name__ == '__main__':
    training_set, testing_set = load_local_data('')

    with tf.compat.v1.Session as sess:
        mnist_cnn = CNN(sess, 28, 28, 'mnist_models')
        mnist_cnn.set_debug_mode(True)
        mnist_cnn.init_classifier()
        mnist_cnn.init_progress_tracker(50)
        mnist_cnn.train(training_set, 100, 1, None, False)
        print(mnist_cnn.evaluate(testing_set, 1))

