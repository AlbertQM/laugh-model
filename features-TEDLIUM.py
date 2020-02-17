import pandas as pd
import numpy as np
import librosa
import os
import glob
from sphfile import SPHFile
import re
import featuresHelper as featHelper
    
# Custom headers to use when parsing the annotations    
annotationsHeaders=["filename", "_", "speakerName", "start", "end"]
# These annotations pickle was created before hand by merging all the annotations files.
# We pre-compute this to save time. (See featuresHelper.mergeAnnotationsToPickle)
annotations = pd.read_pickle("TEDLIUM-full_annotations.pkl")

pathToData = os.path.join(featHelper.BASE_DIR, 'tedlium', 'data')

features = []
# Iterate through each sound file and extract the features 
for index, row in annotations.iterrows():
    file_name =  os.path.join(pathToData,str(row["filename"])) + '.sph'

    if re.search("^S\d+$", row["speakerName"]):
        continue

    # Fragments that are not speech are labeled as follow
    # WadeDavis_2003 1 S140 1313.09 1321.34
    # Thus, we can recognize them by the third column
    # class_label = 'noise' if  re.search("^S\d+$", row["speakerName"]) else "speech"

    class_label = 'speech'
    start = row["start"]
    end = row["end"]
    data = featHelper.extractMFCC(file_name, float(start), float(end))
    features.append([data, class_label])

    if index == 1158:
        break

# Convert into a Panda dataframe 
featuresDF = pd.DataFrame(features, columns=['feature','class_label'])

featHelper.saveFeatures(featuresDF, "TEDLIUM-Speech_only-MFCC_only")
