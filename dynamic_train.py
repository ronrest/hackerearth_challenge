from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import numpy as np
import pickle
import os
import time

from data import id2label, label2id, pickle2obj, obj2pickle, maybe_make_pardir
from viz import train_curves
from model_base import ClassifierModel

from dynamic_data import create_data_dict
from image_processing import random_transformations

# TODO: allow alpha to be entered as an argument for training
# TODO: save and restore evals and global_epoch

# SETTINGS
n_valid = 1000          # Number of samples to set aside for validation set
batch_size = 32
in_shape = [48,48,3]    # Image dimensions [rows, cols, n_chanels] for model input

# SETTINGS
datadir = "/path/to/root/dir"
n_valid = 1000

# ##############################################################################
# DATA
# ##############################################################################
print(("#"*70)+"\n"+"PREPARING DATA"+"\n"+("#"*70))

data = create_data_dict(datadir=datadir)

print("Creating validation split")
data["X_valid"] = data["X_train"][:n_valid]
data["Y_valid"] = data["Y_train"][:n_valid]
data["X_train"] = data["X_train"][n_valid:]
data["Y_train"] = data["Y_train"][n_valid:]

# Information about data shapes
print("DATA SIZES")
print("- X_train: ", len(data["X_train"])) #
print("- X_valid: ", len(data["X_valid"])) #
print("- X_test : ", len(data["X_test"]))  #
print("- Y_train: ", len(data["Y_train"])) #
print("- Y_valid: ", len(data["Y_valid"])) #



# ##############################################################################
# TRAIN
# ##############################################################################




def create_augmentation_func(**kwargs):
    """Args:
        shadow:             (tuple of two floats) (min, max) shadow intensity
        shadow_file:        (str) Path fo image file containing shadow pattern
        shadow_crop_range:  (tuple of two floats) min and max proportion of
                            shadow image to take crop from.
        shadow_crop_range:  ()(default=(0.02, 0.25))
        rotate:             (int)(default=180)
                            Max angle to rotate in each direction
        crop:               (float)(default=0.5)
        lr_flip:            (bool)(default=True)
        tb_flip:            (bool)(default=True)
        brightness:         ()(default=) (std, min, max)
        contrast:           ()(default=) (std, min, max)
        blur:               ()(default=3)
        noise:              ()(default=10)
        resampling:PIL.Image.BICUBIC
    """
    def augmentation_func(X):
        return random_transformations(X=X, **kwargs)
    return augmentation_func

# AUgmentation FUnction
aug_func = create_augmentation_func(
    shadow=(0.5, 0.9),
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=30,
    crop=0.66,
    lr_flip=True,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=2,
    noise=10
    )



# ##############################################################################
#                                                         CREATE AND TRAIN MODEL
# ##############################################################################
def create_and_train_model(model_name, logits_func, data, dynamic=False, alpha=0.01, l2=None, n_epochs=30, batch_size=128, print_every=None, overwrite=False, img_shape=None, augmentation_func=None):
    # Create and Train Model
    # snapshot_file = os.path.join("models", model_name, "snapshots/snapshot.chk")
    # plot_file = os.path.join("models", model_name, "training_plot.png")
    # evals_file = os.path.join("models", model_name, "evals.pickle")

    print("\n"+("#"*70)+"\n"+"MODEL NAME = "+model_name+"\n"+("#"*70)+"\n")

    # Check if the model already exists
    model_dir = os.path.join("models", model_name)
    if os.path.exists(model_dir):
        template = ("="*70) +"\n"+("="*70) +"\n"+(" "*30)+"IMPORTANT!\n"+ ("-"*70)+"\nModel with this name already exists.\n{}\n\n"+("="*70)+"\n"+("="*70)+"\n"
        if overwrite:
            print(template.format("WARNING!!!: YOU ARE IN OVERWRITE MODE\nCompletely deleting the directory associated with the previous model"))
            shutil.rmtree(model_dir)
        else:
            print(template.format("Attempting to re-use existing files"))

    # Ensure the necessary file structures exist
    # maybe_make_pardir(snapshot_file)
    # maybe_make_pardir(plot_file)
    # maybe_make_pardir(evals_file)

    # Create model object
    if dynamic:
        assert img_shape is not None, "Need to feed image shape for dynamic option"
        width, height = img_shape
    else:
        img_shape = list(data["X_train"].shape[1:3])
    n_classes = len(data["id2label"])
    model = ClassifierModel(name=model_name, img_shape=img_shape, n_channels=3, logits_func=logits_func, n_classes=n_classes, dynamic=dynamic, l2=l2)

    # # Load the previsously saved evals
    # if os.path.exists(model.evals_file):
    #     print("relaoding evals data")
    #     model.evals = pickle2obj(evals_file)
    #     model.global_epoch = model.evals["global_epoch"]

    # Train the model
    model.train(data, alpha=alpha, n_epochs=n_epochs, batch_size=batch_size, print_every=print_every, augmentation_func=augmentation_func)

    # # Plot the training curves and save them
    # train_curves(train = model.evals["train_acc"], valid = model.evals["valid_acc"], saveto=plot_file)

    # # Snapshot of evals, and global epoch
    # model.evals["global_epoch"] = model.global_epoch
    # obj2pickle(model.evals, evals_file)

    print("DONE TRAINING")


from architectures import inception_v3_FT_A

# TODO: Create confusion matrix to see common misclassificaions
# create_and_train_model("zobo_02", logits_func=my_architectureZ, data=data, dynamic=True, alpha=0.001, n_epochs=1, batch_size=32, print_every=10, overwrite=True, l2=None, img_shape=(40,40), augmentation_func=aug_func)

# INCEPTION MODEL
create_and_train_model("iv3_ft_A_01", logits_func=inception_v3_FT_A, data=data, dynamic=True, alpha=0.001, n_epochs=1, batch_size=4, print_every=10, overwrite=False, l2=None, img_shape=(299,299), augmentation_func=aug_func)
