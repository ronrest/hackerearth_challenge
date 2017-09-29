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
from pretrained_paths import paths_dict

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


inception_v3_frozen_graph_file = paths_dict["inception_v3_frozen"]
def inception_v3_FT_B(X, n_classes, is_training, regularizer=None):
    x = tf.div(X, 255., name="scale")
    he_init = tf.contrib.keras.initializers.he_normal()
    relu = tf.nn.relu
    dropout = 0.8
    dropout_keep_prob = 1-dropout

    with tf.variable_scope('preprocess') as scope:
        x = tf.div(X, 255., name="scale")

    architecture_arg_scope = tf.contrib.slim.nets.inception.inception_v3_arg_scope
    with tf.contrib.framework.arg_scope(architecture_arg_scope()):
        # LOAD GRAPH_DEF FILE
        graph_file = inception_v3_frozen_graph_file
        which_tensors=["InceptionV3/InceptionV3/Mixed_7c/concat_v2:0"] # 8x8x2048
        trunk_ops = graph_from_graphdef_file(graph_file=graph_file, access_these=which_tensors, remap_input={"input:0": x})
        trunk_final_conv = trunk_ops[0]

        with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=is_training):
            # Final pooling and prediction
            with tf.variable_scope('head'):
                kernel_size = [8,8]
                net = tf.contrib.layers.avg_pool2d(trunk_final_conv, kernel_size, padding='VALID', scope='avg_pool')
                # 1 x 1 x 2048
                net = tf.contrib.layers.dropout(net, keep_prob=dropout_keep_prob, scope='dropout')
                # 2048
                logits = tf.contrib.layers.conv2d(net, n_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_logits')
                logits = tf.squeeze(logits, [1, 2], name='logits')

    return logits


# Dictionary of architectures
arc = {}
arc["my_architectureZ"] = my_architectureZ
arc["inception_v3_FT_B"] = inception_v3_FT_B
