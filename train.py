"""
Running the train.py file assumes you have already created the pickle
files. See the following procedure for how to do this:

    Procedure:
    1. Create data pickle file by running the data.py file.
       Make sure you edit the filepaths at the bottom of that file
       to match where you stored the raw data, and where you want to
       save the processed pickle file.
    2. Edit the PATHS section of this file to reflect where you
       saved the pickle file.
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
#                                                     CREATE MODEL ARCHITECTURES
# ##############################################################################
def my_architectureA(X, n_classes, is_training):
    # Initializers
    he_init = tf.contrib.keras.initializers.he_normal() # He et al 2015 initialization
    xavier_init = tf.contrib.keras.initializers.glorot_normal()

    # Scale images to be 0-1
    x = tf.div(X, 255., name="scale")

    # Convolutional layers
    x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=he_init)
    x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=he_init)
    x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=he_init)
    # dropout = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(0.0))

    # Fully Connected Layers
    x = tf.contrib.layers.flatten(x)
    logits = tf.layers.dense(x, units=n_classes, activation=None, kernel_initializer=xavier_init)

    return logits


# ##############################################################################
#                                                         CREATE AND TRAIN MODEL
# ##############################################################################
# Create and Train Model
# BOOKMARK: Tweak the training schedule
model = ClassifierModel(my_architectureA, in_shape=in_shape, n_classes=n_classes, snapshot_file=snapshot_file)
model.train(data, n_epochs=30, batch_size=128, print_every=1000)

print("DONE TRAINING")
