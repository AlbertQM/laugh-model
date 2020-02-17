import numpy as np
import pandas as pd
import os
import keras
import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import featuresHelper as featHelper

def print_prediction(file_name):
    prediction_feature = featHelper.extractMFCCForTest(file_name) 
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )

# Load model
model = keras.models.load_model('./saved_models/laugh-audio-TEDLIUM.h5')
# Load single picke with features
featuresdf = pd.read_pickle("./features/TEDLIUM-features.pkl")
# Load directory with >1 pickle with features
# files = glob.glob('features/scaled/*.pkl')
# featuresdf = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)


# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

DATA_DIR = os.path.join(featHelper.BASE_DIR, 'SVC', 'data')
filename = os.path.join(featHelper.BASE_DIR, DATA_DIR, 'S0001.wav')

print_prediction(filename)
