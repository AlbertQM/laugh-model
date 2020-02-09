import os
import math
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'SVC', 'data')

# Headers from the sanitized csv
my_cols=['Sample', "original_spk", "gender", "original_time", "type_voc", "start_voc", "end_voc"]
metadata = pd.read_csv('../SVC/labels_sane.txt', names=my_cols, engine='python')

for index, row in metadata.iterrows():

    if index == 0:
        continue

    filename = os.path.join(DATA_DIR,str(row["Sample"])) + '.wav'
    myaudio = AudioSegment.from_wav(filename) 
    start = float(row["start_voc"])
    end = float(row["end_voc"])
    start = math.floor(start*1000)
    end = math.floor(end*1000)
    # Grab only the part we're interested in
    chunk = myaudio[start:end]
    chunk.export('./new-data/{}.wav'.format(index), format='wav')

