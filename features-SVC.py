
import os
import featuresHelper as featHelper

# Custom headers to use when parsing the annotations    
annotationsHeaders=['Sample', "original_spk", "gender", "original_time", "type_voc", "start_voc", "end_voc"]

annotationsFileName = "labels_sane.txt"
dataFolder = "noisy-data"

rootDir = os.path.join(featHelper.BASE_DIR, 'SVC')
pathToAnnotations = os.path.join(rootDir, annotationsFileName)
pathToData = os.path.join(rootDir, dataFolder)

featuresDF = featHelper.getFeatures(annotationsHeaders, 
                                    pathToAnnotations, 
                                    pathToData, 
                                    startHeaderName="start_voc",
                                    endHeaderName="end_voc",
                                    labelHeaderName="type_voc",
                                    audioFormat="wav",
                                    skipHeaders=True)
                                    
featHelper.saveFeatures(featuresDF, "Noisy-SVC")
