from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import numpy as np
import pickle
import os
import time
import shutil  # for removing dirs

# TODO: I think importing from data can be removed
#       Actually, i need to move the pickle functions to dynamic_data
from data import id2label, label2id, pickle2obj, obj2pickle, maybe_make_pardir
from viz import train_curves
from model_base import ClassifierModel

from dynamic_data import create_data_dict, str2file
from image_processing import random_transformations
import argparse

p = argparse.ArgumentParser()
p.add_argument("name", type=str, help="Model Name")
p.add_argument("arc", type=str, help="Model Architecture")
p.add_argument("-d", "--data", type=str, default="data", help="Path to directory containing the data")
p.add_argument("-v", "--n_valid",type=int, default=1000, help="Num samples to set aside for validation set")
p.add_argument("-m", "--max_data", type=int, default=100000000, help="Max number of samples to use from training data. Useful for quickly testing a training reigeme")
p.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
p.add_argument("-a", "--alpha", type=float, default=0.001, help="Learning rate alpha")
p.add_argument("-y", "--dynamic", type=bool, default=True, help="Dynamically load data from raw image files?")
p.add_argument("-n", "--n_epochs", type=int, default=1, help="Number of epochs")
p.add_argument("-p", "--print_every", type=int, default=100, help="How often to print out feedback on training (in number of steps)")
p.add_argument("-l", "--l2", type=float, default=None, help="Amount of L2 to apply")
p.add_argument("-s", "--img_dim", type=int, default=32, help="Size of single dimension of image (assuming square image)")
opt = p.parse_args()



# SETTINGS
# datadir = "/home/ronny/TEMP/hackerearth_deep_learning_challenge/a0409a00-8-dataset_dp"
DATA_DIR = opt.data
MAX_DATA = opt.max_data
N_VALID = opt.n_valid          # Number of samples to set aside for validation set

# ##############################################################################
#                               DATA
# ##############################################################################
print(("#"*70)+"\n"+"PREPARING DATA"+"\n"+("#"*70))

data = create_data_dict(datadir=DATA_DIR)

print("Creating validation split")
data["X_valid"] = data["X_train"][:N_VALID]
data["Y_valid"] = data["Y_train"][:N_VALID]
data["X_train"] = data["X_train"][N_VALID:N_VALID+MAX_DATA]
data["Y_train"] = data["Y_train"][N_VALID:N_VALID+MAX_DATA]

# Information about data shapes
print("DATA SIZES")
print("- X_train: ", len(data["X_train"])) #
print("- X_valid: ", len(data["X_valid"])) #
print("- X_test : ", len(data["X_test"]))  #
print("- Y_train: ", len(data["Y_train"])) #
print("- Y_valid: ", len(data["Y_valid"])) #


# ##############################################################################
#                           DATA AUGMENTATION
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

# Augmentation Function
aug_func = create_augmentation_func(
    shadow=(0.01, 0.7), # (0.5, 0.9)
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=15, #30,
    crop=0.66,
    lr_flip=True,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=1, # 2
    noise=4 #10
    )

# ##############################################################################
#                                                         CREATE AND TRAIN MODEL
# ##############################################################################
def create_and_train_model(model_name, logits_func, data, dynamic=False, alpha=0.01, l2=None, n_epochs=30, batch_size=128, print_every=None, overwrite=False, img_shape=None, augmentation_func=None):
    print("\n"+("#"*70)+"\n"+"MODEL NAME = "+model_name+"\n"+("#"*70)+"\n")
    model_dir = os.path.join("models", model_name)

    # Check if the model already exists
    if os.path.exists(model_dir):
        template = ("="*70) +"\n"+("="*70) +"\n"+(" "*30)+"IMPORTANT!\n"+ ("-"*70)+"\nModel with this name already exists.\n{}\n\n"+("="*70)+"\n"+("="*70)+"\n"
        if overwrite:
            print(template.format("WARNING!!!: YOU ARE IN OVERWRITE MODE\nCompletely deleting the directory associated with the previous model"))
            shutil.rmtree(model_dir)
        else:
            print(template.format("Attempting to re-use existing files"))

    # Create model object
    if dynamic:
        assert img_shape is not None, "Need to feed image shape for dynamic option"
        width, height = img_shape
    else:
        img_shape = list(data["X_train"].shape[1:3])
    n_classes = len(data["id2label"])
    model = ClassifierModel(name=model_name, img_shape=img_shape, n_channels=3, logits_func=logits_func, n_classes=n_classes, dynamic=dynamic, l2=l2)

    # Train the model
    model.train(data, alpha=alpha, n_epochs=n_epochs, batch_size=batch_size, print_every=print_every, augmentation_func=augmentation_func)
    print("DONE TRAINING")



# TODO: Create confusion matrix to see common misclassificaions
# from architectures import my_architectureZ
# create_and_train_model("ZZZZ_deleto", logits_func=my_architectureZ, data=data, dynamic=True, alpha=0.001, n_epochs=3, batch_size=32, print_every=5, overwrite=False, l2=None, img_shape=(40,40), augmentation_func=aug_func)



# create_and_train_model("iv3_ft_A_02", logits_func=inception_v3_FT_A, data=data, dynamic=True, alpha=0.0001, n_epochs=48, batch_size=32, print_every=5, overwrite=False, l2=None, img_shape=(299,299), augmentation_func=aug_func)
# create_and_train_model("iv3_ft_A_03", logits_func=inception_v3_FT_A, data=data, dynamic=True, alpha=0.0001, n_epochs=12, batch_size=4, print_every=30, overwrite=False, l2=None, img_shape=(299,299), augmentation_func=aug_func)
# create_and_train_model("iv3_ft_A_01", logits_func=inception_v3_FT_A, data=data, dynamic=True, alpha=0.001, n_epochs=6, batch_size=4, print_every=30, overwrite=False, l2=None, img_shape=(299,299), augmentation_func=aug_func)
# create_and_train_model("iv3_ft_A_04", logits_func=inception_v3_FT_A, data=data, dynamic=True, alpha=0.001, n_epochs=6, batch_size=32, print_every=5, overwrite=False, l2=None, img_shape=(299,299), augmentation_func=aug_func)

from architectures import arc

create_and_train_model(opt.name,
                       logits_func=arc[opt.arc],
                       data=data,
                       dynamic=opt.dynamic,
                       alpha=opt.alpha,
                       n_epochs=opt.n_epochs,
                       batch_size=opt.batch_size,
                       print_every=opt.print_every,
                       overwrite=False,
                       l2=opt.l2,
                       img_shape=(opt.img_dim, opt.img_dim),
                       augmentation_func=aug_func)


# TODO: Look at this tutorial https://github.com/HackerEarth-Challenges/deep_learning_challenge_one/blob/master/CNN_Starter.ipynb
#  and this one https://github.com/HackerEarth-Challenges/deep_learning_challenge_one/blob/master/keras_vggmodel.ipynb
