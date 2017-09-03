""""
Simply contains the ClassifierModel class. Which contains all the
boilerplate code necessary to Create a tensoerlfow graph, and training
operations.

The important thing to create is a function that takes in the input images
and returns the output logits.

See the docstring in train.py for how this is done.
"""
import tensorflow as tf
import numpy as np
import os
import time

# ==============================================================================
#                                                                    PRETTY_TIME
# ==============================================================================
def pretty_time(t):
    """ Given a time in seconds, returns a string formatted as "HH:MM:SS" """
    t = int(t)
    H, r = divmod(t, 3600)
    M, S = divmod(r, 60)
    return "{:02n}:{:02n}:{:02n}".format(H,M,S)


# ##############################################################################
#                                                          CLASSIFIER MODEL BASE
# ##############################################################################
class ClassifierModel(object):
    def __init__(self, logits_func, in_shape, n_classes, snapshot_file):
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
            self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

            # Body
            # self.logits = self.body(self.X, n_classes, self.is_training)
            self.logits = logits_func(self.X, n_classes, self.is_training)
            _, self.preds = tf.nn.top_k(self.logits, k=1)

            # Loss, Optimizer and trainstep
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y, name="loss"))
            self.optimizer = tf.train.AdamOptimizer(self.alpha, name="optimizer")

            # Handle batch normalization
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
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
    def train(self, data, n_epochs, alpha=0.001, batch_size=32, print_every=10):
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
                    print("="*70, "\nEPOCH {}/{} (GLOBAL_EPOCH: {})        ELAPSED TIME: {}".format(epoch, n_epochs, self.global_epoch, pretty_time(time.time()-t0)),"\n"+("="*70))

                    # Shuffle the data
                    permutation = list(np.random.permutation(n_samples))
                    data["X_train"] = data["X_train"][permutation]
                    data["Y_train"] = data["Y_train"][permutation]

                    # Iterate through each mini-batch
                    for i in range(n_batches):
                        Xbatch = data["X_train"][batch_size*i: batch_size*(i+1)]
                        Ybatch = data["Y_train"][batch_size*i: batch_size*(i+1)]
                        feed_dict = {self.X:Xbatch, self.Y:Ybatch, self.alpha:alpha, self.is_training:True}
                        loss, _ = sess.run([self.loss, self.trainstep], feed_dict=feed_dict)

                        # Print feedback every so often
                        if print_every is not None and (i+1)%print_every==0:
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
                    s = "TR ACC: {: 3.3f} VA ACC: {: 3.3f} TR LOSS: {: 3.5f} VA LOSS: {: 3.5f}\n"
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
