"""
Running the train.py file assumes you have already created the pickle
files. See the following procedure for how to do this:

    Procedure:
    1. Create data pickle file by running the data.py file.
       Make sure you edit the filepaths at the bottom of that file
       to match where you stored the raw data, and where you want to
       save the processed pickle file.
    2. Edit the SETTINGS section of this file to reflect where you
       saved the pickle file.
"""
from __future__ import print_function, division, unicode_literals
import os
import glob
import scipy
from scipy import misc
import numpy as np
from data import id2label, label2id, pickle2obj

# SETTINGS
pickle_file = "/path/to/data_pickle_file.pickle"  # Filepath to the pickled data

# DATA
# Load the pickle file created in the data.py file
data = pickle2obj(pickle_file)




