from __future__ import print_function, division, unicode_literals
# import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

# from new_base import ClassifierModel
from dynamic_data import create_prediction_files_list, id2label, label2id
from new_architectures import arc


import argparse
p = argparse.ArgumentParser()
p.add_argument("name", type=str, help="Model Name")
p.add_argument("--arc", type=str, help="Model Architecture")
p.add_argument("--csv", type=str, help="Path to CSV file containing the image ids to do predictions on.")
p.add_argument("-d", "--data", type=str, default="data", help="Path to SPECIFIC directory containing the prediction images")
p.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
p.add_argument("-y", "--dynamic", type=bool, default=True, help="Dynamically load data from raw image files?")
p.add_argument("-s", "--img_dim", type=int, default=299, help="Size of single dimension of image (assuming square image)")
p.add_argument("--best_metric", type=str, default="valid_acc", help="The metric to use for evaluating best model")
p.add_argument("--use_best", type=bool, default=True, help="Use the 'best' snapshot of the model for prediction")
p.add_argument("--saveto", type=str, default="preds.csv", help="Filepath to save the predictions to as a csv file.")
opt = p.parse_args()


def save_formatted_preds(X, preds, file):
    """
    Given a list full filepaths to images, and a list of predictions for each
    of those images it creates a csv file formatted as:

        image_id,label
        test_1000a, candy
        test_1000b, water
        test_1000c, coffee
        test_1000d, water
        test_1001a, rice
        ...

    And saves it to the desired filepath.
    """
    preds = np.array([id2label[pred] for pred in preds], dtype=np.object)
    image_ids = [os.path.splitext(os.path.basename(f))[0] for f in X]
    df = pd.DataFrame({"image_id" : image_ids, "label": preds})

    # Save a predictions csv file
    df.to_csv(file, index=False)


def predict_on_model(model, X, batch_size=32, use_best=True, saveto="preds.csv"):
    # Predict on the model
    preds = model.prediction(X=X, batch_size=batch_size)
    preds = model.prediction(X=X, batch_size=batch_size, verbose=True, best=use_best)
    print("RECEIVED {} predictions frm model".format(len(preds)))
    if saveto:
        save_formatted_preds(X=X, preds=preds, file=saveto)
    return preds




# Check if the model directory actually exists
model_dir = os.path.join("models", opt.name)
if not os.path.exists(model_dir):
    assert True, "WARNING: This model does not exist"

print("CREATING MODEL")
n_classes = len(id2label)
ModelClass = arc[opt.arc]
model = ModelClass(name=opt.name,
                   pretrained_snapshot=None,
                   img_shape=[opt.img_dim, opt.img_dim],
                   n_channels=3,
                   n_classes=n_classes,
                   dynamic=True,
                   l2=None,
                   best_evals_metric=opt.best_metric)


print("DOING PREDICTIONS")
X = create_prediction_files_list(csv_file=opt.csv, datadir=opt.data)
predict_on_model(model, X, batch_size=opt.batch_size, use_best=opt.use_best, saveto=opt.saveto)
print("- Done!")
