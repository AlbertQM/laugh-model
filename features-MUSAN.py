import pandas as pd
import numpy as np
import os
import librosa
import random

def extract_features(file_name, start, end):
       
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=16000, offset=start, duration=end-start) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as _:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled
    

# Headers from the sanitized csv
my_cols=['Sample']
metadata = pd.read_csv('../MUSAN/musan/speech/us-gov/ANNOTATIONS.txt', names=my_cols, engine='python')
# metadata = pd.read_csv('../MUSAN/musan/noise/sound-bible/ANNOTATIONS.txt', names=my_cols, engine='python')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'MUSAN', 'musan', 'speech', 'us-gov' )
# DATA_DIR = os.path.join(BASE_DIR, 'MUSAN', 'musan', 'noise', 'sound-bible' )

features = []
# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():

    if index == 0:
        continue
 
    file_name =  os.path.join(DATA_DIR,"speech-us-gov-") + row["Sample"] + '.wav'
    # file_name =  os.path.join(DATA_DIR,"noise-sound-bible-") + row["Sample"] + '.wav'

    class_label = "speech"
    # class_label = "noise"
    window = 5
    for start_time in range(10):
    # start = np.random.randint(0,500)
    # end = np.random.randint(0,5)
    # start = np.random.rand()
    # end = 1
        start = start_time * window
        end = (start_time + 1) * window
        data = extract_features(file_name, float(start), float(end))
        
        features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

# Save features
featuresdf.to_pickle("features/MUSAN-speech-features-balanced-scaled.pkl")
# featuresdf.to_pickle("features/MUSAN-noise-features.pkl")

print('Finished feature extraction from ', len(featuresdf), ' files')

# Excluded filename from us-gov (shorter than the 10 min average)
# 0109, 0005, 0130, 0201, 0229, 0210, 0250, 0252, 0159, 0067, 0082
# 0215, 0241, 0020, 0103, 0188, 0111, 0057, 0014, 0090, 0176, 0147

# Excluded from sound-bible (too short) 
# 0006, 0026, 0086, 0002, 0046, 0037, 0050, 0069, 0058