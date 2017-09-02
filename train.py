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
#                                                           SUPPORTING FUNCTIONS
# ##############################################################################
def pretty_time(t):
    """ Given a time in seconds, returns a string formatted as "HH:MM:SS" """
    t = int(t)
    H, r = divmod(t, 3600)
    M, S = divmod(r, 60)
    return "{:02n}:{:02n}:{:02n}".format(H,M,S)


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
#                                                          CLASSIFIER MODEL BASE
# ##############################################################################
class ClassifierModelBase(object):
    def __init__(self, in_shape, n_classes, snapshot_file):
        """ Initializes a Base Classifier Class
            in_shape: [rows, cols, channels]
            n_classes: (int)
            snapshot_file: (str) filepath to save snapshots to
        """
        self.evals = {"train_loss": [],
                      "train_acc": [],
                      "valid_loss": [],
                      "valid_acc": [],
                     }
        self.snapshot_file = snapshot_file
        self.in_shape = in_shape
        self.n_classes = n_classes
        self.global_epoch = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Placeholders for user input
            self.X = tf.placeholder(tf.float32, shape=[None] + in_shape, name="X") # [batch, rows, cols, chanels]
            self.Y = tf.placeholder(tf.int32, shape=[None], name="Y") # [batch]
            self.alpha = tf.placeholder_with_default(0.001, shape=None, name="alpha")
            self.is_training = tf.placeholder_with_default(False, shape=None, name="is_training")

            # Body
            self.logits = self.body(self.X, n_classes, self.is_training)
            _, self.preds = tf.nn.top_k(self.logits, k=1)

            # Loss, Optimizer and trainstep
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y, name="loss"))
            self.optimizer = tf.train.AdamOptimizer(self.alpha, name="optimizer")
            self.trainstep = self.optimizer.minimize(self.loss, name="trainstep")

            # Saver (for saving snapshots)
            self.saver = tf.train.Saver(name="saver")

    # ==========================================================================
    #                                                                       BODY
    # ==========================================================================
    def body(self, X, n_classes, is_training):
        """Override this method in child classes.
           must return pre-activation logits of the output layer
        """
        x = tf.contrib.layers.flatten(X)
        logits = tf.contrib.layers.fully_connected(x, n_classes, activation_fn=None)
        return logits

    # ==========================================================================
    #                                                            INITIALIZE_VARS
    # ==========================================================================
    def initialize_vars(self, session):
        if tf.train.checkpoint_exists(self.snapshot_file):
            try:
                print("Restoring parameters from snapshot")
                self.saver.restore(session, self.snapshot_file)
            except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
                msg = "============================================================\n"\
                      "ERROR IN INITIALIZING VARIABLES FROM SNAPSHOTS FILE\n"\
                      "============================================================\n"\
                      "Something went wrong in  loading  the  parameters  from  the\n"\
                      "snapshot. This is most likely due to changes being  made  to\n"\
                      "the  model,  but  not  changing   the  snapshots  file  path.\n\n"\
                      "Loading from a snapshot requires that  the  model  is  still\n"\
                      "exaclty the same since the last time it was saved.\n\n"\
                      "Either: \n"\
                      "- Use a different snapshots filepath to create new snapshots\n"\
                      "  for this model. \n"\
                      "- or, Delete the old snapshots manually  from  the  computer.\n\n"\
                      "Once you have done that, try again.\n\n"\
                      "See the full printout and traceback above  if  this  did  not\n"\
                      "resolve the issue."
                raise ValueError(str(e) + "\n\n\n" + msg)

        else:
            print("Initializing to new parameter values")
            session.run(tf.global_variables_initializer())

    # ==========================================================================
    #                                                   SAVE_SNAPSHOT_IN_SESSION
    # ==========================================================================
    def save_snapshot_in_session(self, session):
        """Given an open session, it saves a snapshot of the weights to file"""
        # Create the directory structure for parent directory of snapshot file
        if not os.path.exists(os.path.dirname(self.snapshot_file)):
            os.makedirs(os.path.dirname(self.snapshot_file))
        self.saver.save(session, self.snapshot_file)

    # ==========================================================================
    #                                                                      TRAIN
    # ==========================================================================
    def train(self, data, n_epochs, batch_size=32, print_every=100):
        """Trains the model, for n_epochs given a dictionary of data"""
        n_samples = len(data["X_train"])               # Num training samples
        n_batches = int(np.ceil(n_samples/batch_size)) # Num batches per epoch

        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess)
            t0 = time.time()

            try:
                # TODO: Use global epoch
                for epoch in range(1, n_epochs+1):
                    self.global_epoch += 1
                    print("="*70, "\nEPOCH {}/{} (GLOBAL_EPOCH: {})".format(epoch, n_epochs, self.global_epoch),"\n"+("="*70))

                    # Shuffle the data
                    permutation = list(np.random.permutation(n_samples))
                    data["X_train"] = data["X_train"][permutation]
                    data["Y_train"] = data["Y_train"][permutation]

                    # Iterate through each mini-batch
                    for i in range(n_batches):
                        Xbatch = data["X_train"][batch_size*i: batch_size*(i+1)]
                        Ybatch = data["Y_train"][batch_size*i: batch_size*(i+1)]
                        feed_dict = {self.X:Xbatch, self.Y:Ybatch, self.alpha:0.01, self.is_training:True}
                        loss, _ = sess.run([self.loss, self.trainstep], feed_dict=feed_dict)

                        # Print feedback every so often
                        if i%print_every==0:
                            print("{}    Batch_loss: {}".format(pretty_time(time.time()-t0), loss))

                    # Save parameters after each epoch
                    self.save_snapshot_in_session(sess)

                    # Evaluate on full train and validation sets after each epoch
                    train_acc, train_loss = self.evaluate_in_session(data["X_train"], data["Y_train"], sess)
                    valid_acc, valid_loss = self.evaluate_in_session(data["X_valid"], data["Y_valid"], sess)
                    self.evals["train_acc"].append(train_acc)
                    self.evals["train_loss"].append(train_loss)
                    self.evals["valid_acc"].append(valid_acc)
                    self.evals["valid_loss"].append(valid_loss)

                    # Print evaluations
                    s = "\nTR ACC: {: 3.3f} VA ACC: {: 3.3f} TR LOSS: {: 3.5f} VA LOSS: {: 3.5f}\n"
                    print(s.format(train_acc, valid_acc, train_loss, valid_loss))

            except KeyboardInterrupt:
                print("Keyboard Interupt detected")
                # TODO: Finish up gracefully. Maybe create recovery snapshots of model

    # ==========================================================================
    #                                                                 PREDICTION
    # ==========================================================================
    def prediction(self, X):
        """Given input X make a forward pass of the model to get predictions"""
        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess)
            preds = sess.run(self.preds, feed_dict={self.X: X})
            return preds.squeeze()

    # ==========================================================================
    #                                                                   EVALUATE
    # ==========================================================================
    def evaluate(self, X, Y, batch_size=32):
        """Given input X, and Labels Y, evaluate the accuracy of the model"""
        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess)
            return self.evaluate_in_session(X,Y, sess, batch_size=batch_size)


    # ==========================================================================
    #                                                        EVALUATE_IN_SESSION
    # ==========================================================================
    def evaluate_in_session(self, X, Y, session, batch_size=32):
        """ Given input X, and Labels Y, and already open tensorflow session,
            evaluate the accuracy of the model
        """
        # Dimensions
        preds = np.zeros(Y.shape[0], dtype=np.int32)
        loss = 0
        n_samples = Y.shape[0]
        n_batches = int(np.ceil(n_samples/batch_size))

        # Make Predictions on mini batches
        for i in range(n_batches):
            Xbatch = X[batch_size*i: batch_size*(i+1)]
            Ybatch = Y[batch_size*i: batch_size*(i+1)]
            feed_dict = {self.X:Xbatch, self.Y:Ybatch, self.is_training:False}
            batch_preds, batch_loss = session.run([self.preds, self.loss], feed_dict=feed_dict)
            preds[batch_size*i: batch_size*(i+1)] = batch_preds.squeeze()
            loss += batch_loss

        accuracy = (preds.squeeze() == Y.squeeze()).mean()*100
        loss = loss / n_samples
        return accuracy, loss


# ##############################################################################
#                                                               CLASSIFIER MODEL
# ##############################################################################
class ClassifierModel(ClassifierModelBase):
    def __init__(self, in_shape, n_classes, snapshot_file):
        super(ClassifierModel, self).__init__(in_shape, n_classes, snapshot_file)

    def body(self, X, n_classes, is_training):
        # BOOKMARK: This body() method is the most important thing to modify

        # Initializers
        he_init = tf.contrib.keras.initializers.he_normal() # He et al 2015 initialization
        xavier_init = tf.contrib.keras.initializers.glorot_normal()

        # dropout = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(0.0))

        # Scale images to be 0-1
        x = tf.div(X, 255., name="scale")

        # Convolutional layers
        x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=he_init)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=he_init)
        x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=he_init)

        # x = tf.contrib.layers.conv2d(x, num_outputs=8, kernel_size=3, stride=2, padding="SAME")
        # x = tf.contrib.layers.conv2d(x, num_outputs=16, kernel_size=3, stride=2, padding="SAME")
        # x = tf.contrib.layers.conv2d(x, num_outputs=32, kernel_size=3, stride=2, padding="SAME")

        # Deconvolutional layers
        #x = conv2d_transpose(x, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=None, name="DC1") #### Deconvolutional Layer

        # Fully Connected Layers
        x = tf.contrib.layers.flatten(x)
        logits = tf.layers.dense(x, units=n_classes, activation=None, kernel_initializer=xavier_init)
        # logits = tf.contrib.layers.fully_connected(x, n_classes, activation_fn=None)

        return logits


# ##############################################################################
#                                                         CREATE AND TRAIN MODEL
# ##############################################################################
# Create and Train Model
# BOOKMARK: Tweak the training schedule
model = ClassifierModel(in_shape=in_shape, n_classes=n_classes, snapshot_file=snapshot_file)
model.train(data, n_epochs=5, batch_size=128, print_every=100)

