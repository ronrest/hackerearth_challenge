from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import numpy as np
import pickle
import os
import time

from data import id2label, label2id, pickle2obj, obj2pickle, maybe_make_pardir
from viz import train_curves
from model_base import ClassifierModel, create_and_train_model

from dynamic_data import create_data_dict
from image_processing import random_transformations

# TODO: allow alpha to be entered as an argument for training
# TODO: save and restore evals and global_epoch

# SETTINGS
n_valid = 1000          # Number of samples to set aside for validation set
batch_size = 32
in_shape = [48,48,3]    # Image dimensions [rows, cols, n_chanels] for model input

# SETTINGS
datadir = "/path/to/root/dir"
n_valid = 1000

# ##############################################################################
# DATA
# ##############################################################################
data = create_data_dict(datadir=datadir)

print("Creating validation split")
data["X_valid"] = data["X_train"][:n_valid]
data["Y_valid"] = data["Y_train"][:n_valid]
data["X_train"] = data["X_train"][n_valid:]
data["Y_train"] = data["Y_train"][n_valid:]

# Information about data shapes
print("DATA SIZES")
print("- X_train: ", len(data["X_train"])) #
print("- X_valid: ", len(data["X_valid"])) #
print("- X_test : ", len(data["X_test"]))  #
print("- Y_train: ", len(data["Y_train"])) #
print("- Y_valid: ", len(data["Y_valid"])) #



# ##############################################################################
# TRAIN
# ##############################################################################

def my_architectureZ(X, n_classes, is_training, regularizer=None):
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





def create_augmentation_func(**kwargs):
    """Args:
        shadow:             (tuple of two floats) (min, max) shadow intensity
        shadow_file:        (str) Path fo image file containing shadow pattern
        shadow_crop_range:  (tuple of two floats) min and max proportion of
                            shadow image to take crop from.
        shadow_crop_range:  ()(default=(0.02, 0.25))
        rotate:             (int)(default=180)
                            Max angle to rotate in each direction
        crop:               (float)(default=0.5)
        lr_flip:            (bool)(default=True)
        tb_flip:            (bool)(default=True)
        brightness:         ()(default=) (std, min, max)
        contrast:           ()(default=) (std, min, max)
        blur:               ()(default=3)
        noise:              ()(default=10)
        resampling:PIL.Image.BICUBIC
    """
    def augmentation_func(X):
        return random_transformations(X=X, **kwargs)
    return augmentation_func

# AUgmentation FUnction
aug_func = create_augmentation_func(
    shadow=(0.5, 0.9),
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=30,
    crop=0.66,
    lr_flip=True,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=2,
    noise=10
    )


# TODO: Create confusion matrix to see common misclassificaions
create_and_train_model("zobo_02", logits_func=my_architectureZ, data=data, dynamic=True, alpha=0.001, n_epochs=4, batch_size=32, print_every=10, overwrite=True, l2=None, img_shape=(40,40), augmentation_func=aug_func)
