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

        my_logits_function(X, n_classes, is_training, regularizer=None)

    NOTE: You can call the function whatever you want, as long as the arguments
    all conform to the same API.

    X = a tensorflow placeholder tensor that will be passed to your function.
        This will contain the current batch of images for you to perform a
        forward pass of the neural network.

    n_classes = an integer specifying the number of output classes for
        the final output layer

    is_training = Another tensorflow placeholder that conatains a boolean value.
        This tells your architecture if the model is currently in training mode
        (True), or evaluation/production/prediction mode (False).

    regularizer = A tensorflow regularizer object that is passed to your
        function.
            NOTE: this object is only passed to your function if
            you declare an L2 value in the `create_and_train_model()`
            function mentioned below (if no L2 value is used, then
            no regularizer object will be passed to your function,
            which is why a default value of None needs to be included in
            the API)

    RETURN:
    The function should return a tensorflow tensor of your final output layer
    (the logits just before performing a softmax operation).

2. Call the `create_and_train_model()` function:

        create_and_train_model(
            model_name="myModelA",
            logits_func=my_logits_function,
            data=data,
            alpha=0.01,
            n_epochs=20,
            batch_size=32,
            print_every=10,
            overwrite=False,
            l2=None)


    The important things to modify are:
    - model_name = A name to give your model. NOTE: Treat any slight changes
        in parameters as completely new models.
        This name, will create a subdirectory inside the "models" directory.
        - Here, the model's weights will be saved in snapshot files.
        - an "evals" file will be saved here to which contains information
          about the history of its performance over time.
        - An image containing how the model performed on training and validation
          data after each epoch will be saved here.
    - logits_function = the function you created above that returns the logits
    - alpha = the learning rate
    - n_epochs = the number of epochs to run training for
    - batch_size = Size of the minibatches
    - print_every = Controls how often it gives feedback in between epochs.
        Set this to `None` if you only want it to print out feedback
        after each epoch,but nothing in between.
    - overwrite = If set to True, and it encounters a model with the same
        name already saved, then it will delete EVERYTHING from the previous
        model, and train this model from scratch.

        If set to False, and it encounters a previously saved model with the
        same name, then it will try to continue training from where the
        previous model left off. This will only work if the architecture is
        exactly the same as the previously saved model.
    - l2 = Se this to None if you are not going to use regularization.
        Otherwise, set it to some float, that specifies the amount of L2
        regularization to apply to your model.

        NOTE: This does not automatically apply regularization. It simply
        passes a tensorflow regularizer object to your logits function.
        you will need to actually manually specify which layers you want to
        apply that regularizer to in your architecture.

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

from data import id2label, label2id, pickle2obj, obj2pickle, maybe_make_pardir
from viz import train_curves
from base import ClassifierModel, create_and_train_model

# TODO: allow alpha to be entered as an argument for training
# TODO: save and restore evals and global_epoch

# SETTINGS
n_valid = 1000          # Number of samples to set aside for validation set
batch_size = 32
in_shape = [32,32,3]    # Image dimensions [rows, cols, n_chanels] for model input


# PATHS
pickle_file = "/path/to/data_pickle_file.pickle"  # Filepath to the pickled data
pickle_file = "data.pickle"

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
def my_architectureA(X, n_classes, is_training, regularizer=None):
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

create_and_train_model("modelA", logits_func=my_architectureA, data=data, alpha=0.01, n_epochs=20, batch_size=32, print_every=10, overwrite=True, l2=None)
