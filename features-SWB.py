import pandas as pd
import os
import featuresHelper as featHelper

def sanitizeSampleName(sampleName):
  # We need to go from "sw2005A-ms98-a-penn.text" to "sw02005"
  # First, split by "-", and take the first part ("sw2005A")
  sanitizedSampleName = sampleName.split("-")[0]
  # Then, we need to remove the last char ("sw2005")
  sanitizedSampleName = sanitizedSampleName[:-1]
  # Then, add a 0 after "sw" ("sw02005")
  sanitizedSampleName = "sw0" + sanitizedSampleName[2:]
  return sanitizedSampleName

def getSampleClass(annotation):
  if annotation.startswith("[silence"):
    return "silence"
  elif annotation.startswith("[laughter"):
    return "laugh"
  # Skip annotation adjustments
  elif annotation == "---" or annotation == "+++":
    return None
  else:
    return "speech"

# Custom headers to use when parsing the annotations    
annotationsHeaders=["Sample", "original_spk", "start_voc", "end_voc", "modifier", "_oldAnnotation", "annotation"]

annotationsFileName = "swb-full-annotations.pkl"
dataFolder = "swb1"

rootDir = os.path.join(featHelper.BASE_DIR, 'switchboard')
pathToAnnotations = os.path.join(rootDir, annotationsFileName)
pathToData = os.path.join(rootDir, dataFolder)

metadata = pd.read_pickle(pathToAnnotations)

features = []
for index, row in metadata.iterrows():
  sampleName = str(row["Sample"])
  fileName =  os.path.join(pathToData, sanitizeSampleName(sampleName)) + '.sph'
  annotation = getSampleClass(row['annotation'])
  if annotation == None:
    continue
  start = row["start_voc"]
  end = row["end_voc"]
  data = featHelper.extractAll(fileName, float(start), float(end))
  features.append([data, annotation])

# Convert into a Pandas dataframe 
featuresDf = pd.DataFrame(features, columns=['feature','classLabel'])
featHelper.saveFeatures(featuresDf, "SWB-43")
