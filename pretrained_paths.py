"""
A really horrible hacky way to get the other python scripts to find
model file paths on different computers.

Each computer will have a different version of this file.
"""

from computer import NAME

if NAME == "ronny":
    paths_dict = {
        "inception_v3_frozen": "/home/ronny/TEMP/pretrained_models/tfslim/inception/inception_v3/graph/inception_v3_2016_08_28_frozen.pb",
    }
elif NAME == "aws":
    paths_dict = {
        "inception_v3_frozen": "/home/ubuntu/efs/pretrained/inceptionv3_frozen/inception_v3_2016_08_28_frozen.pb",
    }
