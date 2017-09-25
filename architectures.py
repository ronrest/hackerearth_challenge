from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import tensorflow.contrib.slim.nets
import numpy as np
import pickle
import os
import time

from data import id2label, label2id, pickle2obj, obj2pickle, maybe_make_pardir
from viz import train_curves
from base import ClassifierModel, create_and_train_model
from model_base import graph_from_graphdef_file



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


