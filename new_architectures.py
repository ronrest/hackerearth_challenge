
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets

from new_base import ClassifierModel



class InceptionV3_Pretrained(ClassifierModel):
    def __init__(self, name, pretrained_snapshot, img_shape, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_acc"):
        super().__init__(name=name, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

        self.pretrained_snapshot = pretrained_snapshot
        print("CREATED model with snapshot: ", self.pretrained_snapshot)


    def create_body_ops(self):
        """Override this method in child classes.
           must return pre-activation logits of the output layer

           Ops to make use of:
               self.is_training
               self.X
               self.Y
               self.alpha
               self.dropout
               self.l2_scale
               self.l2
        """

        with tf.variable_scope('preprocess') as scope:
            self.scaled_inputs = tf.div(self.X, 255., name="scale")

        # IMPORT INCEPTION v3
        # INCEPTION MODEL
        # This arg_scope is absolutely essential!
        # Without it, it creates the layers incorectly
        arg_scope = tf.contrib.slim.nets.inception.inception_v3_arg_scope
        architecture_func = tf.contrib.slim.nets.inception.inception_v3
        with tf.contrib.framework.arg_scope(arg_scope()):
            self.logits, _ = architecture_func(self.scaled_inputs,
                                        num_classes=self.n_classes,
                                        is_training=self.is_training,
                                        dropout_keep_prob=0.8)

        self.preds = tf.argmax(self.logits, axis=-1, name="preds")


    def create_saver_ops(self):
        """ Create operations to save/restore model weights """
        with tf.device('/cpu:0'): # prevent more than one thread doing file I/O
            # Inception Saver
            excluded_weights = ["InceptionV3/AuxLogits", "InceptionV3/Logits"]
            trunk_vars = tf.contrib.framework.get_variables_to_restore(include=["InceptionV3"], exclude=excluded_weights)
            self.pretrained_saver = tf.train.Saver(trunk_vars, name="trunk_saver")

            # Main Saver
            main_vars = tf.contrib.framework.get_variables_to_restore(exclude=None)
            self.saver = tf.train.Saver(main_vars, name="saver")


    def initialize_vars(self, session, best=False):
        # INITIALIZE VARS
        if best:
            main_snapshot_file = self.best_snapshot_file
        else:
            main_snapshot_file = self.snapshot_file

        with tf.device('/cpu:0'):
            if tf.train.checkpoint_exists(main_snapshot_file):
                print(" Loading from Main chekcpoint")
                self.saver.restore(session, main_snapshot_file)
            else:
                print("Initializing from pretrained inception")
                print("- file: ", self.pretrained_snapshot)
                session.run(tf.global_variables_initializer())
                self.pretrained_saver.restore(session, self.pretrained_snapshot)

        # TODO: Do error handling, eg:
        #
        # """ Override this if you set up custom savers """
        # if best:
        #     snapshot_file = self.best_snapshot_file
        # else:
        #     snapshot_file = self.snapshot_file
        # if tf.train.checkpoint_exists(snapshot_file):
        #     try:
        #         print("Restoring parameters from snapshot")
        #         self.saver.restore(session, snapshot_file)
        #     except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
        #         msg = "============================================================\n"\
        #               "ERROR IN INITIALIZING VARIABLES FROM SNAPSHOTS FILE\n"\
        #               "============================================================\n"\
        #               "Something went wrong in  loading  the  parameters  from  the\n"\
        #               "snapshot. This is most likely due to changes being  made  to\n"\
        #               "the  model,  but  not  changing   the  snapshots  file  path.\n\n"\
        #               "Loading from a snapshot requires that  the  model  is  still\n"\
        #               "exaclty the same since the last time it was saved.\n\n"\
        #               "Either: \n"\
        #               "- Use a different snapshots filepath to create new snapshots\n"\
        #               "  for this model. \n"\
        #               "- or, Delete the old snapshots manually  from  the  computer.\n\n"\
        #               "Once you have done that, try again.\n\n"\
        #               "See the full printout and traceback above  if  this  did  not\n"\
        #               "resolve the issue."
        #         raise ValueError(str(e) + "\n\n\n" + msg)
        #
        # else:
        #     print("Initializing to new parameter values")
        #     session.run(tf.global_variables_initializer())


arc={}
arc["InceptionV3_Pretrained"] = InceptionV3_Pretrained
