import os
import glob
import scipy
from scipy import misc
import numpy as np

datadir = "/path/to/root/dir"
pattern = "*.png"
img_shape = [32,32] # Shape to resize the images to
n_channels = 3      # Number of channels to use # TODO: implement converting to greyscale
n_valid = 1000       # Number of samples to set aside for validation

train_files = glob.glob(os.path.join(datadir, "train_img", pattern))
test_files = glob.glob(os.path.join(datadir, "test_img", pattern))

data = {}
data["X_train"] = np.zeros([n_samples]+img_shape+[n_channels])
data["Y_train"] = []
data["X_valid"] = []
data["Y_valid"] = []
data["X_test"] = []
data["Y_test"] = []

for dataset_name, file_list in [("train", train_files), ("test", test_files)]:
    n_samples = len(file_list)

    data["X_"+dataset_name] = np.zeros([n_samples]+img_shape+[n_channels])

    # TODO: Load up labels
    # data["Y_"+dataset_name] = np.zeros([n_samples])

    # # Small subset durung development to keep it easy to debug
    # # TODO: Remove these two lines
    # n_samples = 10
    # file_list = file_list[:n_samples]

    for i, file in enumerate(file_list):
        img_file = file_list[i]
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
print("X_train: ", data["X_train"].shape) # ('X_train: ', (2215, 32, 32, 3))
print("X_valid: ", data["X_valid"].shape) # ('X_valid: ', (1000, 32, 32, 3))
print("X_test : ", data["X_test"].shape)  # ('X_test : ', (1732, 32, 32, 3))


