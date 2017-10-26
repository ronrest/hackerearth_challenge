## Description

Using an inception v3 model, pretrained on imagenet data, and fine tuned for the
task of classifying grocery items.

Model was created in tensorflow.

## Environment

- ubuntu 16.04
- Python 3.5 in a python virtualenv with the  following python libraries installed:

```
graphviz==0.8
h5py==2.7.1
image-geometry==1.11.15
interactive-markers==1.11.3
ipykernel==4.6.1
ipython==6.1.0
ipython-genutils==0.2.0
ipywidgets==7.0.0
jedi==0.10.2
jmespath==0.9.0
jsonschema==2.6.0
jupyter==1.0.0
jupyter-client==5.1.0
jupyter-console==5.2.0
jupyter-core==4.3.0
language-selector==0.1
laser-geometry==1.6.4
louis==2.5.3
lxml==3.3.3
Markdown==2.6.9
MarkupSafe==1.0
matplotlib==1.5.1
nbconvert==5.3.1
nbformat==4.4.0
notebook==5.0.0
numpy==1.13.1
pandas==0.18.1
pandocfilters==1.4.2
pathspec==0.3.4
pexpect==4.2.1
pickleshare==0.7.4
Pillow==4.1.0
piston-mini-client==0.7.5
pluginlib==1.10.5
prompt-toolkit==1.0.15
protobuf==3.4.0
psutil==4.2.0
pycrypto==2.6.1
pycurl==7.19.3
pydot==1.2.3
python-magic==0.4.13
pytz==2016.4
pyzmq==16.0.2
rviz==1.11.17
scipy==0.17.0
six==1.10.0
tensorflow==1.3.0
tensorflow-tensorboard==0.1.6
```

# Data

The data is downloaded from [here](https://he-s3.s3.amazonaws.com/media/hackathon/deep-learning-challenge-1/identify-the-objects/a0409a00-8-dataset_dp.zip)

The data is extracted, and the path to the extracted directory is stored as a
variable.

```sh
DATA_DIR="/home/ubuntu/efs/hackerearth_data"
```


## Pretrained weights

The model makes use of pretrained weights that are publically available from the
Tensorflow project.

We specify the directory we want to save the weights to and move into that
directory.

```sh
WEIGHTS_DIR="/home/ubuntu/efs/pretrained/inceptionv3"
mkdir -p ${WEIGHTS_DIR}
cd ${WEIGHTS_DIR}
```

We now download the weights

```sh
wget -c http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xzvf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
```

We will want to access the weights file, so we store the path as a variable.

```sh
PRETRAINED_CHECKPOINT_FILE=${WEIGHTS_DIR}/inception_v3.ckpt
```



## Setting up the project for training

We specify the directory that contains the source code files. And move into
that directory.

```sh
PROJECT_DIR="/home/ubuntu/efs/hackerearth_challenge"
cd ${PROJECT_DIR}
```

Next we run the following commands, which send variables to the python scripts
to train on an inception v3 model. It will create a subdirectory called
`models`, which will conatin another subdirectory with the model name, which
will contain all the files associated with the model being trained.

This training reigeme is the one that was used in the submission.

```sh
# START TRAINING
NAME="aws_inception_v3_TR_A_03"
ARC="InceptionV3_Pretrained"
N_EPOCHS=89
BATCH_SIZE=32
ALPHA=0.00001
python new_train.py $NAME --arc $ARC -d $DATA_DIR --pretrained_snapshot $PRETRAINED_CHECKPOINT_FILE -n $N_EPOCHS -b $BATCH_SIZE -s 299 -a $ALPHA -p 5 -v 100 -y True

N_EPOCHS=11
ALPHA=0.000003
python new_train.py $NAME --arc $ARC -d $DATA_DIR --pretrained_snapshot $PRETRAINED_CHECKPOINT_FILE -n $N_EPOCHS -b $BATCH_SIZE -s 299 -a $ALPHA -p 5 -v 100 -y True

N_EPOCHS=10
ALPHA=0.000001
python new_train.py $NAME --arc $ARC -d $DATA_DIR --pretrained_snapshot $PRETRAINED_CHECKPOINT_FILE -n $N_EPOCHS -b $BATCH_SIZE -s 299 -a $ALPHA -p 5 -v 100 -y True
```


## Setting up the project for predictions

Once the model is trained, we can perform predictions.

Just specify the path to the csv file containing the image ids to be used for
prediction, the specific directory containing the images for prediction, and
where we want the predictions to be saved.


```sh
PREDICTION_IDS_FILE="${DATA_DIR}/test.csv" # The csv file containing the image ids for predictions
PREDICTIONS_DIR="${DATA_DIR}/test_img"     # The specific dir containing images for predictions
PREDICTIONS_OUTPUT="preds.csv"             # WHere to save the predictions
```

Now we run the following code to get it to do the predictions and save the
predictions to the csv file we specified above.

```sh
# MAKE PREDICTOINS
USE_BEST=True  # whether to make use of the best snapshot for the model
python new_predict.py $NAME --arc $ARC --csv $PREDICTION_IDS_FILE -d $PREDICTIONS_DIR -b 32 -s 299 -y True --use_best $USE_BEST --saveto $PREDICTIONS_OUTPUT
```
