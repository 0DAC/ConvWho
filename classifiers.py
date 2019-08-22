""" Contains the CNN class and basic run variables """

import os
# make tf shut up
# specific info
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random as r

# for local print statements
debug = False

model_name = "human_classifier"
data_path = '../labelled-face-data'
model_path = "../" + model_name + "-cnn/"
spacer = "------------------------------------------------------------------------------"

try:
    os.makedirs(model_path)
except FileExistsError:
    # directory already exists
    pass

# Image input parameters
image_height = 200
image_width = 200
color_channels = 3
gender = {
    0: 'Male',
    1: 'Female'
}

ethnicity = {
        0: 'White',
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Others (includes Hispanic, Latino, Middle Eastern)'
    }

# Training parameters
training_steps = 100
batch_size = 64
load_checkpoint = True

class ConvNet:

    def __init__(self, image_height, image_width, channels, num_classes):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels],
                                              name="inputs")
            smrtprint(self.input_layer.shape)

            conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same",
                                            activation=tf.nn.relu)
            smrtprint(conv_layer_1.shape)

            pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
            smrtprint(pooling_layer_1.shape)

            conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=64, kernel_size=[5, 5], padding="same",
                                            activation=tf.nn.relu)
            smrtprint(conv_layer_2.shape)

            pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
            smrtprint(pooling_layer_2.shape)

            flattened_pooling = tf.layers.flatten(pooling_layer_2)
            dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
            smrtprint(dense_layer.shape)
            dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
            outputs = tf.layers.dense(dropout, num_classes)
            smrtprint(outputs.shape)

            self.choice = tf.argmax(outputs, axis=1)
            self.probability = tf.nn.softmax(outputs)

            self.labels = tf.placeholder(dtype=tf.float32, name="labels")
            self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)

            one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())
            self.saver = tf.train.Saver(max_to_keep=2)

def smrtprint(args):
    if debug: print(args)

def process_data(data):

    smrtprint("Processing Data")

    smrtprint("\t>>> Normalizing values to [0, 1]")
    float_data = np.array(data, dtype=float) / 255.0

    smrtprint("\t>>> Enforcing reshaping into 200x200")
    reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))

    smrtprint("\t>>> Transposing data")
    transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])

    smrtprint("Transposing Complete")

    return transposed_data

def parse_data(file_path, max_files_to_load=-1):
    """
    Extracts labels from test dataset
    source: https://susanqq.github.io/UTKFace/

    :param file_path: path to data
    :return: array data and dicts age_labels, gender_labels, ethnicity_labels
    """

    dir = os.fsdecode(file_path)

    data = []
    age_labels = []
    gender_labels = []
    ethnicity_labels = []

    print("Parsing data from %s" % file_path)
    l = len(os.listdir(dir))
    counter = 0
    print_progress_bar(counter, l, prefix='Progress:', suffix='Complete', length=50)
    for file in os.listdir(dir):
        if counter == max_files_to_load:
            break
        filename = os.fsdecode(file)
        img = cv2.imread(file_path + filename, cv2.IMREAD_UNCHANGED)

        if filename.startswith('.'):
            counter+=1
            continue

        filename = filename.split('.')[0]

        try:
            age, gender, race, date_time = filename.split('_')
            age = eval(age)
            gender = eval(gender)
            race = eval(race)
            data.append(img)
            age_labels.append(age)
            gender_labels.append(gender)
            ethnicity_labels.append(race)
            # print("Age: %d, Gender: %s, Ethnicity: %s" % (eval(age), 'female' if gender else 'male', ethnicity[eval(race)]))
            print_progress_bar(counter, l if max_files_to_load==-1 else max_files_to_load, prefix='Progress:', suffix='Complete', length=50)
            counter += 1
        except:
            # print("Dude, fucked up again on %s" % filename)
            continue
    print("Complete")
    return data, age_labels, gender_labels, ethnicity_labels

def synth_tf_dataset(data, labels):
    """
    Generates tf iterable dataset and its iterator

    :param data: raw data values
    :param labels: labels for data
    :return: dataset, dataset_iterator
    """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=labels.shape[0])
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    dataset_iterator = dataset.make_initializable_iterator()

    return dataset, dataset_iterator

def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s (%s/%s)' % (prefix, bar, percent, suffix, iteration, total), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def space():
    print(spacer)

if __name__ == '__main__':
    gender_cnn = ConvNet(image_height, image_width, color_channels, 2)
    age_cnn = ConvNet(image_height, image_width, color_channels, 116)
    race_cnn = ConvNet(image_height, image_width, color_channels, 5)

    # parse data
    train_data, train_age_labels, train_gender_labels, train_race_labels = parse_data(data_path + '/train/')
    eval_data, eval_age_labels, eval_gender_labels, eval_race_labels = parse_data(data_path + '/test/')

    train_labels = {
        'age':train_age_labels,
        'gender':train_gender_labels,
        'race':train_race_labels
    }

    eval_labels = {
        'age':eval_age_labels,
        'gender':eval_gender_labels,
        'race':eval_race_labels
    }

    # process data
    train_data = process_data(train_data)
    eval_data = process_data(eval_data)

    # batch-erize it
    train_batches = {'age':[],
                     'gender': [],
                     'race': []}
    test_batches = {'age':[],
                     'gender': [],
                     'race': []}
    accuracies = []
    # iterate through age, gender, and race labels
    for metric in train_labels:

        space()
        print("Processing training data for %s" % metric)

        space()
        print("Splicing data")
        for batch in range(0, len(train_data), batch_size):
            print("\t>>> Brewing batch %d" % (batch//batch_size+1))
            train_batches[metric].append(np.asarray(train_labels[metric][batch:batch+batch_size]))
            # print("Currently at image %s" % (batch))
        print("Splicing complete")
        space()

        print("Metric: %s" % metric)
        if metric == 'age':
            cnn = age_cnn
        elif metric == 'gender':
            cnn = gender_cnn
        elif metric == 'race':
            cnn = race_cnn
        else:
            print("\tError: No CNN exists for %s" % metric)
            space()
            break
        saver = cnn.saver

        # iterate through each batch
        for n, batch in enumerate(train_batches[metric]):
            tf.reset_default_graph()
            print("Processing batch %s of %s" % (n+1, len(train_batches[metric])))

            print("\t>>> Opening TF session")
            with tf.Session(graph=cnn.graph) as sess:
                dataset, iterator = synth_tf_dataset(train_data[n * batch_size:(n + 1) * batch_size],
                                                     train_batches[metric][n])
                next_element = iterator.get_next()
                if load_checkpoint:
                    print("\t>>> Loading checkpoint")
                    checkpoint = tf.train.get_checkpoint_state(model_path + "/" + metric + "/")
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                    print("\t>>> Checkpoint restored")
                else:
                    print("\t>>> No existing checkpoint, initializing global variables")
                    sess.run(tf.global_variables_initializer())

                sess.run(tf.local_variables_initializer())
                sess.run(iterator.initializer)

                print("\t>>> Initiating training")
                for step in range(training_steps):
                    print("\t\t[Step %d]" % step)
                    current_batch = sess.run(next_element)

                    batch_inputs = current_batch[0]
                    batch_labels = current_batch[1]

                    sess.run((cnn.train_operation, cnn.accuracy_op),
                             feed_dict={cnn.input_layer: batch_inputs, cnn.labels: batch_labels})

                    if step % 9 == 0 and step > 0:
                        current_acc = sess.run(cnn.accuracy)
                        accuracies.append(current_acc)
                        p_change = (accuracies[len(accuracies)-2]-current_acc)/accuracies[len(accuracies)-2]*100

                        print("Accuracy at step " + str(step) + ": " + str(current_acc))
                        print("%d percent %s " % (abs(p_change), "increase" if p_change < 0 else "decrease"))
                        print("Saving checkpoint")
                        saver.save(sess, model_path + "/" + metric + "/", step)

                print("Saving final checkpoint for training session.")
                load_checkpoint=True

    # train the model
    for metric in eval_labels:
        print("Metric: %s" % metric)
        if metric == 'age':
            cnn = age_cnn
        elif metric == 'gender':
            cnn = gender_cnn
        elif metric == 'race':
            cnn = race_cnn
        else:
            print("\tError: No CNN exists for %s" % metric)
            space()
            break
        saver = cnn.saver
        with tf.Session(graph=cnn.graph) as sess:
            checkpoint = tf.train.get_checkpoint_state(model_path + "/" + metric + "/")
            saver.restore(sess, checkpoint.model_checkpoint_path)

            indexes = np.random.choice(len(eval_data), 10, replace=False)
            rows = 5
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
            fig.patch.set_facecolor('white')
            image_count = 0

            for idx in indexes:
                image_count += 1
                sub = plt.subplot(rows, cols, image_count)
                img = eval_data[idx]
                img = img.reshape(28, 28)
                plt.imshow(img)
                guess = sess.run(cnn.choice, feed_dict={cnn.input_layer: [eval_data[idx]]})
                guess_name = eval_labels[metric][guess[0]]
                actual_name = eval_labels[metric][eval_labels[idx]]
                sub.set_title("G: " + guess_name + " A: " + actual_name)
            plt.tight_layout()
            plt.show()