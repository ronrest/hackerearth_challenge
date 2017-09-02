"""
            FUNCTIONS FOR PROCESSING THE DATA

If you run this file directly from the command line, it will generate the
pickled data. Modify the settings at the very botton of this file under
    if __name__ == '__main__'
to configure how the pickled files get generated.

Otherwise, you can import the indivial variables and functions to other python
files.

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
        if not os.path.exists(path):
            os.makedirs(path)


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, filepath):
    """ Saves a python object as a binary pickle file to the desired filepath"""
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
def create_data_pickle(datadir, pickle_file, label2id, img_shape=[32,32], n_valid=1000):
    """ Given the root directory that contains the raw data, it
        saves a pickle file that contains a python dictionary
        with the following keys:

            "X_train", "Y_train", "X_valid", "Y_valid", "X_test"

        The "X_train", "X_valid", and "X_test" items contain numpy
        arrays storing the images with the following dimensions:

            [n_batch, img_rows, img_cols, n_channels]

        The "Y_train", Y_valid" items are numpy arrays containing
        the integer ids for the labels, in the following dimensions:

            [n_batch, 1]
    """
    print("CREATING DATA PICKLES")
    n_channels = 3      # Number of channels to use

    # DATA
    # Data keys will be  "X_train", "Y_train", "X_valid", "Y_valid", "X_test"
    data = {}

    # Train files and labels
    print("- Extracting train filenames and labels")
    labels_file = os.path.join(datadir, "train.csv")
    train_labels = np.genfromtxt(labels_file, delimiter=",", skip_header=1, dtype=str)
    train_files = train_labels[:,0]
    train_labels = train_labels[:,1]
    data["Y_train"] = np.array(list(map(lambda label: label2id[label], train_labels))).astype(np.int8)
    data["Y_train"] = data["Y_train"].reshape(-1, 1) # reshape to column vector

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
        data["X_"+dataset_name] = np.zeros([n_samples]+img_shape+[n_channels], dtype=np.int8)
        for i, filename in enumerate(file_list):
            img_file = os.path.join(datadir, dataset_name+"_img", filename+".png")
            img = scipy.misc.imread(img_file)

            # ---------------------
            # Preprocess the images
            # ---------------------
            img = scipy.misc.imresize(img, img_shape) # resize

            # Add the processed image to the array
            data["X_"+dataset_name][i] = img

    # Create Validation data (from first `n_valid` samples of training set)
    print("- Creating validation split")
    # TODO: Check that the distributions of labels has been preserved in split.
    #       Maybe shuffle before splitting
    data["X_valid"] = data["X_train"][:n_valid]
    data["Y_valid"] = data["Y_train"][:n_valid]
    data["X_train"] = data["X_train"][n_valid:]
    data["Y_train"] = data["Y_train"][n_valid:]


    # Information about data shapes
    print("- DATA SHAPES")
    print("- X_train: ", data["X_train"].shape) # 'X_train: ', (2215, 32, 32, 3)
    print("- X_valid: ", data["X_valid"].shape) # 'X_valid: ', (1000, 32, 32, 3)
    print("- X_test : ", data["X_test"].shape)  # 'X_test : ', (1732, 32, 32, 3)
    print("- Y_train: ", data["Y_train"].shape) # 'X_train: ', (2215,)
    print("- Y_valid: ", data["Y_valid"].shape) # 'X_valid: ', (1000,)

    print("- DATA TYPES")
    print("- X_train: ", data["X_train"].dtype) # 'X_train: ', int8
    print("- X_valid: ", data["X_valid"].dtype) # 'X_valid: ', int8
    print("- X_test : ", data["X_test"].dtype)  # 'X_test : ', int8
    print("- Y_train: ", data["Y_train"].dtype) # 'X_train: ', int8
    print("- Y_valid: ", data["Y_valid"].dtype) # 'X_valid: ', int8

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
    n_valid = 1000      # Number of samples to set aside for validation

    # Create the pickled data
    create_data_pickle(datadir, pickle_file, label2id=label2id,
        img_shape=img_shape, n_valid=n_valid)
