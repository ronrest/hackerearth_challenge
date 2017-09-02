import os
import glob
import scipy
from scipy import misc
import numpy as np

# SETTINGS
datadir = "/path/to/root/dir"
pattern = "*.png"
img_shape = [32,32] # Shape to resize the images to
n_channels = 3      # Number of channels to use # TODO: implement converting to greyscale
n_valid = 1000       # Number of samples to set aside for validation


# MAP LABELS - Between string labels to integer ids
id2label = ['beans', 'cake', 'candy', 'cereal', 'chips', 'chocolate', 'coffee',
    'corn', 'fish', 'flour', 'honey', 'jam', 'juice', 'milk', 'nuts',
    'oil', 'pasta', 'rice', 'soda', 'spices', 'sugar', 'tea',
    'tomatosauce', 'vinegar', 'water']
label2id = {val:id for id,val in enumerate(id2label)}

# DATA
data = {}
# Data keys will be  "X_train", "Y_train", "X_valid", "Y_valid", "X_test"

# Train files and labels
labels_file = os.path.join(datadir, "train.csv")
train_labels = np.genfromtxt(labels_file, delimiter=",", skip_header=1, dtype=str)
train_files = labels[:,0]
train_labels = labels[:,1]
data["Y_train"] = np.array(list(map(lambda label: label2id[label], train_labels)))

# Test files
labels_file = os.path.join(datadir, "test.csv")
test_files = np.genfromtxt(labels_file, delimiter=",", skip_header=1, dtype=str)

# LOAD THE DATA
for dataset_name, file_list in [("train", train_files), ("test", test_files)]:
    n_samples = len(file_list)

    # # Small subset during development to keep it easy to debug
    # # TODO: Remove these two lines
    # n_samples = 10
    # file_list = file_list[:n_samples]

    data["X_"+dataset_name] = np.zeros([n_samples]+img_shape+[n_channels])

    if dataset_name == "train":
        data["Y_train"] = np.zeros([n_samples])

    # TODO: Load up labels
    # data["Y_"+dataset_name] = np.zeros([n_samples])

    for i, filename in enumerate(file_list):
        img_file = os.path.join(datadir, dataset_name+"_img", filename+".png")
        img = scipy.misc.imread(img_file)
        img = scipy.misc.imresize(img, img_shape)
        data["X_"+dataset_name][i] = img


# Create Validation data (from first `n_valid` samples of training set)
# TODO: Check that the distributions of labels has been preserved in split.
#       Maybe shuffle before splitting
data["X_valid"] = data["X_train"][:n_valid]
data["Y_valid"] = data["Y_train"][:n_valid]
data["X_train"] = data["X_train"][n_valid:]
data["Y_train"] = data["Y_train"][n_valid:]


# Information about data shapes
print("DATA SHAPES")
print("X_train: ", data["X_train"].shape) # 'X_train: ', (2215, 32, 32, 3)
print("X_valid: ", data["X_valid"].shape) # 'X_valid: ', (1000, 32, 32, 3)
print("X_test : ", data["X_test"].shape)  # 'X_test : ', (1732, 32, 32, 3)
print("Y_train: ", data["Y_train"].shape) # 'X_train: ', (2215,)
print("Y_valid: ", data["Y_valid"].shape) # 'X_valid: ', (1000,)


# TODO: Pickle the processed data to make it faster to load
# NOTE: Use protocol=2 when pickling

