from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical, np_utils
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
import glob
import numpy as np
import pandas as pd

# featuresdf = pd.read_pickle("./features.pkl")

# Concat all pickles - Containing features extracted from different datasets
files = glob.glob('features/scaled/*.pkl')
featuresdf = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
print(yy[0], '<---- x0')
print(y[0], '<---- y0')
print(yy[-1]), '<------ x-1'
print(y[-1], ' <------y-1')
# split the dataset 
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

num_labels = yy.shape[1]
filter_size = 3

# Construct model 
model = Sequential()

FEATURES = 40
FIRST_LSTM_CELLS = 90
SECOND_LSTM_CELLS = 60

model.add(Dense(FIRST_LSTM_CELLS, input_shape=(FEATURES,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(SECOND_LSTM_CELLS))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_vad.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

model.save('./saved_models/laugh-audio-vad-v2.h5')

# # Convert model to tfjs
# # tensorflowjs_converter --input_format keras saved_models/laugh-audio-vad.h5 ./vad-v2