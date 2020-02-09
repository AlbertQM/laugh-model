import pandas as pd
import numpy as np
import librosa
import os

NUM_MFCCS = 20
SIGNAL_FREQ = 16000

def extract_features(file_name, start, end):
       
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=SIGNAL_FREQ, offset=start, duration=end-start) 
        # Extracts an array of length NUM_MFCCS, where each array is length 9 (num of frames).
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NUM_MFCCS)
        # Take the 9 values and average them, ending up with a flat array of length 40
        mfccsScaled = np.mean(mfccs.T, axis=0)
        # mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        # Compute the spectrogram and take the first 50 magnitudes
        melSpectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)[:50]
        melSpectrogramScaled = np.mean(melSpectrogram.T, axis=0)
        features = np.concatenate((melSpectrogramScaled, mfccsScaled), axis=0)
        delta = librosa.feature.delta(features)
    except Exception as _:
        print("Error encountered while parsing file: ", file_name)
        print('\n', _, '\n')
        return None 
     
    return np.concatenate((features, delta), axis=0)
    
# Headers from the sanitized csv
my_cols=['Sample', "original_spk", "gender", "original_time", "type_voc", "start_voc", "end_voc"]
metadata = pd.read_csv('../SVC/labels_sane.txt', names=my_cols, engine='python')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'SVC', 'data')

features = []
# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():

    if index == 0:
        continue
    
    file_name =  os.path.join(DATA_DIR,str(row["Sample"])) + '.wav'
    
    class_label = row["type_voc"]
    start = row["start_voc"]
    end = row["end_voc"]
    data = extract_features(file_name, float(start), float(end))
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

# Save features
featuresdf.to_pickle("features/SVC-features-full.pkl") # Consider HDF5 in alternative to Pickle
print('Finished feature extraction from ', len(featuresdf), ' files')