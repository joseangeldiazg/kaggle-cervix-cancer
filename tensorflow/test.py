from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_floatx('float32')
import time
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import os
from keras.models import model_from_json

test = glob.glob("pruebaTest/*.jpg")
test = pd.DataFrame([[p[11:len(p)],p] for p in test], columns = ['image','path'])
test_data = normalize_image_features(test['path'])
np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)
test_id = test.image.values
np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)

experiment_name = "Experimentos/fine"

json_file = open(experiment_name + '/model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)

model.load_weights(experiment_name + "/CAMBIARMANUALMENTEPORELMEJOR")

pred = model.predict(test_data)

df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
df.to_csv('submission.csv', index=False)
