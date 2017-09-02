"""
################################################################################
             USING THIS FILE FOR PROCESSING THE DATA
################################################################################
If you run this file directly from the command line, it will generate the
pickled data necessary to start training the models.

To do this, scroll down to the very bottom and  modify the following settings

- datadir = the file path to the root directory of the hackerearth data.
            It should be the directory containing:

            test_img/
            train_img/
            sample_submission.csv
            test.csv
            train.csv

- pickle_file = specify a filepath to save the processed images as.
                This will be a "pickle" file, which is just a python
                dictionary object saved on your computer to make it quicker
                to load the data during training.

                The dictionary object stored contains the following keys:
                "X_train", "Y_train", "X_test"
                And each of those contains a numpy array.

- img_shape =   The 2D dimensions to reshape all the images to.
                eg: [32,32]
                Start with something small like [32,32], to experiment with,
                then only move to something bigger if none of the models
                you try are performing well enough in terms of accuracy.


################################################################################
                       OTHER USES FOR THIS FILE
################################################################################
This file contains useful functions for processing files which are useful for
the other python files. Simply import the function you want using:

    from data import some_function

################################################################################
"""
from __future__ import print_function, unicode_literals
import os
import glob
import scipy
from scipy import misc
import numpy as np
import pickle


# MAP LABELS - Between string labels to integer ids
id2label = ['beans', 'cake', 'candy', 'cereal', 'chips', 'chocolate', 'coffee',
    'corn', 'fish', 'flour', 'honey', 'jam', 'juice', 'milk', 'nuts',
    'oil', 'pasta', 'rice', 'soda', 'spices', 'sugar', 'tea',
    'tomatosauce', 'vinegar', 'water']
label2id = {val:id for id,val in enumerate(id2label)}


# ==============================================================================
#                                                                 MAYBE_MAKE_DIR
# ==============================================================================
def maybe_make_dir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, filepath):
    """ Saves a python object as a binary pickle file to the desired filepath"""
    maybe_make_pardir(filepath) # Ensure parent director exists
    with open(filepath, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=2)


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Given a filepath to a pickle file it returns the python object inside"""
    with open(file, mode = "rb") as fileObj:
        obj = pickle.load(fileObj)
    return obj


# ==============================================================================
#                                                             CREATE_DATA_PICKLE
# ==============================================================================
def create_data_pickle(datadir, pickle_file, label2id, img_shape=[32,32]):
    """ Given the root directory that contains the raw data, it
        saves a pickle file that contains a python dictionary
        with the following keys:

            "X_train", "Y_train", "X_test"

        The "X_train" and "X_test" items contain numpy
        arrays storing the images with the following dimensions:

            [n_batch, img_rows, img_cols, n_channels]

        The "Y_train" item is a numpy array containing
        the integer ids for the labels, in the following dimensions:

            [n_batch]
    """
    print("CREATING DATA PICKLES")
    n_channels = 3      # Number of channels to use

    # DATA
    # Data keys will be  "X_train", "Y_train", "X_test"
    data = {}

    # Train files and labels
    print("- Extracting train filenames and labels")
    labels_file = os.path.join(datadir, "train.csv")
    train_labels = np.genfromtxt(labels_file, delimiter=",", skip_header=1, dtype=str)
    train_files = train_labels[:,0]
    train_labels = train_labels[:,1]
    data["Y_train"] = np.array(list(map(lambda label: label2id[label], train_labels))).astype(np.uint8)
    data["Y_train"] = data["Y_train"].reshape(-1) # ensure dimension is [n_batches]

    # Test files
    print("- Extracting test filenames")
    labels_file = os.path.join(datadir, "test.csv")
    test_files = np.genfromtxt(labels_file, delimiter=",", skip_header=1, dtype=str)

    # LOAD THE DATA
    for dataset_name, file_list in [("train", train_files), ("test", test_files)]:
        print("- Processing the {} image files".format(dataset_name))
        n_samples = len(file_list)

        # # Small subset during development to keep it easy to debug
        # # TODO: Remove these two lines
        # n_samples = 10
        # file_list = file_list[:n_samples]

        # Populate each image at a time into the dataset
        data["X_"+dataset_name] = np.zeros([n_samples]+img_shape+[n_channels], dtype=np.uint8)
        for i, filename in enumerate(file_list):
            img_file = os.path.join(datadir, dataset_name+"_img", filename+".png")
            img = scipy.misc.imread(img_file)

            # ---------------------
            # Preprocess the images
            # ---------------------
            img = scipy.misc.imresize(img, img_shape) # resize

            # Add the processed image to the array
            data["X_"+dataset_name][i] = img

    # # Create Validation data (from first `n_valid` samples of training set)
    # print("- Creating validation split")
    # # TODO: Check that the distributions of labels has been preserved in split.
    # #       Maybe shuffle before splitting
    # data["X_valid"] = data["X_train"][:n_valid]
    # data["Y_valid"] = data["Y_train"][:n_valid]
    # data["X_train"] = data["X_train"][n_valid:]
    # data["Y_train"] = data["Y_train"][n_valid:]
    #

    # Information about the data
    print("- DATA KEYS: ", data.keys())
    print("- DATA SHAPES")
    print("- X_train: ", data["X_train"].shape) # 'X_train: ', (3215,, 32, 32, 3)
    print("- X_test : ", data["X_test"].shape)  # 'X_test : ', (1732, 32, 32, 3)
    print("- Y_train: ", data["Y_train"].shape) # 'Y_train: ', (3215,)

    print("- DATA TYPES")
    print("- X_train: ", data["X_train"].dtype) # 'X_train: ', uint8
    print("- X_test : ", data["X_test"].dtype)  # 'X_test : ', uint8
    print("- Y_train: ", data["Y_train"].dtype) # 'X_train: ', uint8

    # Save the pickle file
    print("- Pickling the data to file")
    obj2pickle(data, pickle_file)
    print("- DONE")


if __name__ == '__main__':
    # SETTINGS
    datadir = "/path/to/root/dir"
    pickle_file = "/path/to/data_pickle_file.pickle"
    img_shape = [32,32] # Shape to resize the images to
    n_channels = 3      # Number of channels to use

    # Create the pickled data
    create_data_pickle(datadir, pickle_file, label2id=label2id, img_shape=img_shape)
