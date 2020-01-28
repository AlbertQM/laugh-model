import numpy as np
import pandas as pd
import os
import keras
import librosa
import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

model = keras.models.load_model('./saved_models/laugh-audio-librivox-svc-v2.h5')
# featuresdf = pd.read_pickle("./features.pkl")
files = glob.glob('features/scaled/*.pkl')
featuresdf = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)


# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name, e)
        return None, None

    return np.array([mfccsscaled])

    
def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'SVC', 'data')
filename = os.path.join(BASE_DIR, DATA_DIR, 'S0001.wav')
print_prediction('./test_laugh.wav')
