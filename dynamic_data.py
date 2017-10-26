"""
Dynamically loads images from file.


Data zip file: https://he-s3.s3.amazonaws.com/media/hackathon/deep-learning-challenge-1/identify-the-objects/a0409a00-8-dataset_dp.zip

"""
from __future__ import print_function, unicode_literals
import os
import glob
import scipy
import scipy.misc
import numpy as np

# MAP LABELS - Between string labels to integer ids
id2label = ['beans', 'cake', 'candy', 'cereal', 'chips', 'chocolate', 'coffee',
    'corn', 'fish', 'flour', 'honey', 'jam', 'juice', 'milk', 'nuts',
    'oil', 'pasta', 'rice', 'soda', 'spices', 'sugar', 'tea',
    'tomatosauce', 'vinegar', 'water']
label2id = {val:id for id,val in enumerate(id2label)}


# ==============================================================================
#                                                                 MAYBE_MAKE_DIR
# ==============================================================================
# Code taken from: https://github.com/ronrest/convenience_py/blob/master/file/maybe_make_dir.py
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
# Code taken from: https://github.com/ronrest/convenience_py/blob/master/file/maybe_make_dir.py
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
# Code taken from: https://github.com/ronrest/convenience_py/blob/master/file/pickle.py
def obj2pickle(obj, filepath):
    """ Saves a python object as a binary pickle file to the desired filepath"""
    maybe_make_pardir(filepath) # Ensure parent director exists
    with open(filepath, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=2)


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
# Code taken from: https://github.com/ronrest/convenience_py/blob/master/file/pickle.py
def pickle2obj(file):
    """ Given a filepath to a pickle file it returns the python object inside"""
    with open(file, mode = "rb") as fileObj:
        obj = pickle.load(fileObj)
    return obj


# ==============================================================================
#                                                                       STR2FILE
# ==============================================================================
# Code taken from : https://github.com/ronrest/convenience_py/blob/master/file/str2file.py
def str2file(s, file, mode="w"):
    with open(file, mode=mode) as fileObj:
        fileObj.write(s)


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_data_dict(datadir):
    """ Given the root directory that contains the raw data, it
        returns a dictionary with the following keys:

            "X_train", "Y_train", "X_test", "id2label"
            "label2id", "n_classes"

        Where:

            X_train: Full file paths to training images
            Y_train: Id indices for each class
            X_test:  Full file paths to test images
            id2label:
            label2id:
            n_classes:
    """
    print("Creating Data Dictionary")
    # Data keys will be  "X_train", "Y_train", "X_test"
    data = {}

    # Label and id mappings
    data["id2label"] = id2label
    data["label2id"] = {val:id for id,val in enumerate(id2label)}
    data["n_classes"] = len(id2label)

    # TRAIN DATA
    print("- Extracting train filenames and labels")
    csv_file = os.path.join(datadir, "train.csv")
    csv_data = np.genfromtxt(csv_file, delimiter=",", skip_header=1, dtype=str)
    train_files = csv_data[:,0]
    train_labels = csv_data[:,1]

    # Full file paths to training images
    data["X_train"] = [os.path.join(datadir, "train_img", filename+".png") for filename in train_files]
    data["X_train"] = np.array(data["X_train"], dtype=np.object)

    # Id indices for each class
    data["Y_train"] = np.array(list(map(lambda label: label2id[label], train_labels))).astype(np.uint8)
    data["Y_train"] = data["Y_train"].reshape(-1) # ensure dimension is [n_batches]

    # TEST FILES
    print("- Extracting test filenames")
    # Full file paths to test images
    csv_file = os.path.join(datadir, "test.csv")
    test_files = np.genfromtxt(csv_file, delimiter=",", skip_header=1, dtype=str)
    data["X_test"] = [os.path.join(datadir, "test_img", filename+".png") for filename in test_files]
    data["X_test"] = np.array(data["X_test"], dtype=np.object)

    print("- Done")
    return data


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_prediction_files_list(csv_file, datadir):
    """ Given a filepath to a CSV file with the image ids, and the path to
        the SPECIFIC directory that contains the images you want to
        do predictions on, it returns a
        list of the full file paths to all the images:
    """
    print("Creating prediction files list")

    # TEST FILES
    files = np.genfromtxt(csv_file, delimiter=",", skip_header=1, dtype=str)
    data = [os.path.join(datadir, filename+".png") for filename in files]
    data = np.array(data, dtype=np.object)

    print("- Done")
    return data


# ==============================================================================
#                                                           LOAD_BATCH_OF_IMAGES
# ==============================================================================
# Code taken from: https://github.com/ronrest/convenience_py/blob/master/ml/img/load_images.py
def load_batch_of_images(file_list, img_shape):
    """ Given a list of file images to load, it loads them as an array.
    Args:
        file_list:
        img_shape: (tuple of two ints)(width,height)
    Return:
        Numpy array of shape [n_images, img_shape[1], img_shape[0], n_channels]
    """
    n_channels = 3      # Number of channels to use
    n_samples = len(file_list)
    width, height = img_shape
    images = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)

    # Populate each image at a time into the dataset
    for i, img_file in enumerate(file_list):
        img = scipy.misc.imread(img_file)

        # PROCESS THE IMAGES
        img = scipy.misc.imresize(img, img_shape) # resize

        # Add the processed image to the array
        images[i] = img

    return images


if __name__ == '__main__':
    # # USAGE
    # datadir = "/path/to/root/dir"
    # data = create_data_dict(datadir=datadir)
    # X_batch = load_batch_of_images(data["X_train"][:10])
    pass
