# Load various imports 
import pandas as pd
import numpy as np
import os
import librosa

# Headers from the sanitized csv
my_cols=['Sample', "original_spk", "gender", "original_time", "type_voc", "start_voc", "end_voc"]
metadata = pd.read_csv('../SVC/labels_sane.txt', names=my_cols, engine='python')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'SVC', 'data')

def extract_features(file_name, start, end):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=16000, offset=start, duration=end-start) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as _:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled

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
featuresdf.to_pickle("./features.pkl") # Consider HDF5 in alternative to Pickle
print('Finished feature extraction from ', len(featuresdf), ' files')