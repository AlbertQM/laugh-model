from pydub import AudioSegment
import glob
import librosa
import numpy as np
import os
import pandas as pd
import random


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_MFCCS = 40
SAMPLE_RATE = 16000

def extractMFCC(fileName, start, end):
    """Extract Mel-Ceptral Coefficients from audio signal."""
    try:
        audio, sample_rate = librosa.load(fileName, res_type='kaiser_fast', sr=SAMPLE_RATE, offset=start, duration=end - start) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NUM_MFCCS)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as _:
        print("Error encountered while parsing file: ", fileName)
        return None 
     
    return mfccsscaled

def extractMFCCForTest(fileName):
    """Extract MFCCs from a single file. Used for testing the model."""
    try:
        audio, sample_rate = librosa.load(fileName, res_type='kaiser_fast', sr=SAMPLE_RATE) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NUM_MFCCS)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as _:
        print("Error encountered while parsing file: ", fileName)
        return None 
     
    return np.array([mfccsscaled])

def extractAll(file_name, start, end):
    """Extracts MFCCs, ZCR, Spectral flatness and Spectral centroid"""
    try:
        # Load the righ part of the sample
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=SAMPLE_RATE, offset=start, duration=end-start) 
        # Extracts an array of length NUM_MFCCS, where each array is length 9 (num of frames).
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NUM_MFCCS)
        # Take the 9 values and average them, ending up with a flat array of length NUM_MFCCS
        mfccsScaled = np.mean(mfccs.T, axis=0)
        
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcrScaled = np.mean(zcr.T, axis=0)

        spectralFlatness = librosa.feature.spectral_flatness(y=audio)
        spectralFlatnessScaled = np.mean(spectralFlatness.T, axis=0)

        spectralCentroid =  librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectralCentroidScaled = np.mean(spectralCentroid.T, axis=0)
        
        # Merge all features
        features = np.concatenate((mfccsScaled, zcrScaled, spectralCentroidScaled, spectralFlatnessScaled), axis=0)
    except Exception as _:
        print("Error encountered while parsing file: ", file_name)
        print('\n', _, '\n')
        return None 

    return features

def getFeatures(annotationsHeaders, annotationsPath, dataPath, startHeaderName, endHeaderName, labelHeaderName, audioFormat = 'wav', skipHeaders = False, manualLabelOverride = False):
    """Given annotations and data, iterate through the the former and extract features from the latter.

        Parameters
        ----------
        annotationsHeaders : str[]
            The list of the custom headers of the annotations
        annotationsPath : str
            Full path to the directory containing the annotation file
        dataPath : int, optional
            Full path to the directory containing the data files
        audioFormat: string
            The format of the audio file, e.g. "wav", "sph"
        skipHeaders : boolean, default False
            Whether to skip the first line when reading annotations
        startHeaderName : string
            Name of the header/column which contains where in time to begin extracting features
        endHeaderName : string
            Name of the header/column which contains where in time to end extracting features
        labelHeaderName : string
            Name of the header/column which contains the label of the sample
        manualLabelOverride : string, optional, default False
            Assign a label manually rather than reading it from the annotations                                    
            """
    
    metadata = pd.read_csv(annotationsPath, names=annotationsHeaders, engine='python')

    features = []
    for index, row in metadata.iterrows():

        if skipHeaders and index == 0:
            continue

        fileName =  os.path.join(dataPath,str(row["Sample"])) + '.' + audioFormat
        classLabel = row[labelHeaderName] if manualLabelOverride == False else manualLabelOverride
        start = row[startHeaderName]
        end = row[endHeaderName]
        data = extractAll(fileName, float(start), float(end))
        features.append([data, classLabel])

    # Convert into a Panda dataframe 
    return pd.DataFrame(features, columns=['feature','classLabel'])

def saveFeatures(features, fileName):
    """ Store extracted features in a pickle file.

    Parameters
    ----------        
    features : Pandas DataFrame
        Assign a label manually rather than reading it from the annotations
    fileName: string
        Name of the picke that will be created   
    
    """
    features.to_pickle("features/" + fileName + ".pkl")

def createNoisyDataset():
    # whiteNoise = AudioSegment.from_file( os.path.join(BASE_DIR, 'SVC', 'noise', 'white-noise') + '.wav')
    # crowd = AudioSegment.from_file( os.path.join(BASE_DIR, 'SVC', 'noise', 'crowd') + '.wav')
    # bgNoise = AudioSegment.from_file( os.path.join(BASE_DIR, 'SVC', 'noise', 'bg-noise') + '.wav')
    # windNoise = AudioSegment.from_file( os.path.join(BASE_DIR, 'SVC', 'noise', 'wind-noise') + '.wav')
    # bgOutdoors = AudioSegment.from_file( os.path.join(BASE_DIR, 'SVC', 'noise', 'bg-outdoors') + '.wav')
    # bgMedium = AudioSegment.from_file( os.path.join(BASE_DIR, 'SVC', 'noise', 'bg-medium') + '.wav')
    # noises = [whiteNoise, crowd, bgMedium, bgNoise, bgOutdoors, windNoise]
    ## Create noisy dataset
    # sample = AudioSegment.from_file(file_name)
    # combined = sample.overlay(random.choice(noises))
    # combined.export("./noisy-SVC/" + str(row["Sample"]) + ".wav", format='wav')
    print('Not implemented')

def mergeAnnotationsToPickle(pathToFolderOfAnnotations, annotationsHeaders, separator, fileName):
    """ Merge >1 annotation files and stores them in a single pickle.

    Parameters
    ----------        
     pathToFolderOfAnnotations : string
        Path to the folder containing >1 annotations files you wish to merge
    annotationsHeaders : str[]
        The list of the custom headers of the annotations      
    separator : str or RegExpt
        Separator to use when reading the annotations
    fileName: string
        Name of the picke that will be created        
    """
    all_files = glob.glob(pathToFolderOfAnnotations + '*')
    dfFromFiles = (pd.read_csv(f, names=annotationsHeaders, engine='python', sep=separator) for f in all_files)
    fullDf = pd.concat(dfFromFiles, ignore_index=True)
    fullDf.to_pickle(fileName + '.pkl')
