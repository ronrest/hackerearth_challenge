"""
################################################################################
#                               NOTES ABOUT DATA
################################################################################
Running the train.py file assumes you have already created the pickle
file that contains the processed images. See the docstring at the top
of the data.py file for instructions on how to create the pickle file.

################################################################################
#                                 NOTES ABOUT MODEL
################################################################################
I (Ronny) have created a python class caled `ClassifierModel` that already
includes most of the boilerplate code necessary for creating a tensorflow
project that will go through all the training process.

The main thing that needs to be done is to experiment with different model
architectures. To create a new model architecture, simply do the following:

1.  create your own function with the following API:

        my_logits_function(X, n_classes, is_training)

    X = a tensorflow placeholder tensor that will be passed to your function.
        This will contain the current batch of images for you to perform a
        forward pass of the neural network.

    n_classes = an integer specifying the number of output classes for
        the final output layer

    is_training = Another tensorflow placeholder that conatains a boolean value.
        This tells your architecture if the model is currently in training mode
        (True), or evaluation/production/prediction mode (False).


    The function should return a tensorflow tensor of your final output layer
    (the logits just before performing a softmax operation).

2.  Create an instance of `ClassifierModel` that passes the function you created
    above as the first argument.

    The full API is as follows:

        mymodel = ClassifierModel(my_logits_function, in_shape=[32,32,3], n_classes=25, snapshot_file="path/to/snapshot_file")

    my_logits_function = the function you created above that returns the logits

    in_shape = The dimensions of the input images [n_rows,n_cols,n_channels]

    n_classes = The number of output classes to clsasify

    snapshot_file = Name of file to save the model weights to.


3.  Call the train method:

        mymodel.train(data, alpha=0.01, n_epochs=30, batch_size=128, print_every=10)

    alpha = learning rate for training

    data = the data dictionary contianing "X_train", "Y_train", "X_valid", "Y_valid", "X_test"

    n_epochs = The number of epochs to train for.

    batch_size = How big each mini-batch should be

    print_every = Controls how often it gives feedback in between epochs.
        How many steps should elapse before giving feedback?
        Set this to `None` if you only want it to print out feedback
        after each epoch,but nothing in between.

################################################################################
#                                 THINGS TO TRY
################################################################################
1.  Different architectures
2.  Different image sizes (control this in the data.py file to control how big
    the images will be in the  pickled data)
3.  Data augmentation steps.
    - this can be done with either built in tensorflow functions for image
      processing, and apply these in your logits_function.
    - Or, grater range of image processing steps could be performed by
      editing the train() method in the ClassifierModel class to do image
      processing on the batch of images before they get passed to your
      logits_function.
################################################################################
"""
from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import numpy as np
import pickle
import os
import time

from data import id2label, label2id, pickle2obj, maybe_make_pardir
from viz import train_curves
from base import ClassifierModel

# TODO: allow alpha to be entered as an argument for training
# TODO: save and restore evals and global_epoch

# SETTINGS
n_valid = 1000          # Number of samples to set aside for validation set
batch_size = 32
in_shape = [32,32,3]    # Image dimensions [rows, cols, n_chanels] for model input


# PATHS
pickle_file = "/path/to/data_pickle_file.pickle"  # Filepath to the pickled data
pickle_file = "data.pickle"
snapshot_file = "snapshots/snapshot.chk"
plot_file = "plots/training_plot.png" # Saves a plot of the training curves

# Ensure the necessary file structures exist
maybe_make_pardir(snapshot_file)
maybe_make_pardir(plot_file)

n_classes = len(id2label)
print("Number of classes = ", n_classes) # 25


# ##############################################################################
#                                                                DATA PROCESSING
# ##############################################################################
# DATA - Load the pickle file created in the data.py file
# Data keys will be  "X_train", "Y_train", "X_valid", "Y_valid", "X_test"
data = pickle2obj(pickle_file)

# Create Validation data (from first `n_valid` samples of training set)
print("Creating validation split")
# TODO: Check that the distributions of labels has been preserved in split.
#       Maybe shuffle before splitting
data["X_valid"] = data["X_train"][:n_valid]
data["Y_valid"] = data["Y_train"][:n_valid]
data["X_train"] = data["X_train"][n_valid:]
data["Y_train"] = data["Y_train"][n_valid:]

# Information about data shapes
print("DATA SHAPES")
print("- X_train: ", data["X_train"].shape) # - X_train:  (2215, 32, 32, 3)
print("- X_valid: ", data["X_valid"].shape) # - X_valid:  (1000, 32, 32, 3)
print("- X_test : ", data["X_test"].shape)  # - X_test :  (1732, 32, 32, 3)
print("- Y_train: ", data["Y_train"].shape) # - Y_train:  (2215,)
print("- Y_valid: ", data["Y_valid"].shape) # - Y_valid:  (1000,)


# ##############################################################################
#                                                 CREATE THE MODEL ARCHITECTURES
# ##############################################################################
def my_architectureA(X, n_classes, is_training):
    # Initializers
    he_init = tf.contrib.keras.initializers.he_normal() # He et al initialization
    xavier_init = tf.contrib.keras.initializers.glorot_normal()
    relu = tf.nn.relu
    dropout = 0.2

    # PREPROCESS INPUTS
    # Scale images to be 0-1
    x = tf.div(X, 255., name="scale")

    # CONVOLUTIONAL LAYERS
    # Conv1
    x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=2, padding='same', activation=relu, kernel_initializer=he_init)
    x = tf.layers.dropout(x,rate=dropout,training=is_training)

    # Conv2
    x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=relu, kernel_initializer=he_init)
    x = tf.layers.dropout(x,rate=dropout,training=is_training)

    # Conv3
    x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='same', activation=relu, kernel_initializer=he_init)
    x = tf.layers.dropout(x,rate=dropout,training=is_training)

    # FULLY CONNECTED LAYERS
    x = tf.contrib.layers.flatten(x)
    logits = tf.layers.dense(x, units=n_classes, activation=None, kernel_initializer=xavier_init)

    return logits


# ##############################################################################
#                                                         CREATE AND TRAIN MODEL
# ##############################################################################
# Create and Train Model
model = ClassifierModel(my_architectureA, in_shape=in_shape, n_classes=n_classes, snapshot_file=snapshot_file)
model.train(data, alpha=0.01, n_epochs=30, batch_size=32, print_every=10)

print("DONE TRAINING")
