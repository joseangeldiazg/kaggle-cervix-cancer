# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from multiprocessing import Pool, cpu_count
from subprocess import check_output
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
K.set_floatx('float32')


def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

def main():
    os.chdir('D:\Facultad\Master\Segundo cuatrimestre\SIGE\PracticaFinal')

    train= glob.glob("prueba1/**/*.png")+glob.glob("prueba2/**/*.jpg")
    train = pd.DataFrame([[p[8:14],p[15:len(p)],p] for p in train], columns = ['type','image','path'])
    train = im_stats(train)
    train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
    train_data = normalize_image_features(train['path'])
    np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)
    le = LabelEncoder()
    train_target = le.fit_transform(train['type'].values)
    np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)

    x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.25, random_state=14)

    datagen = ImageDataGenerator(r
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    datagen.fit(train_data)

    base_model = ResNet50(weights='imagenet', include_top=False)

     # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(base_model.output)
    # add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    # add a logistic layer
    output = Dense(3, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy')
    model.summary()
    model_json = model.to_json()

    experimento_path="/Experimentos/fine"

    if not os.path.exists(experimento_path):
		os.makedirs(experimento_path)

	with open(experiment_name + "/" + "model.json","w") as json_file:
		json_file.write(model_json)

    print("Entrenando...")

    filepath= + "/checkpoint-val_loss{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(generator=datagen.flow(x_train, y_train,
                        batch_size=5, shuffle=True),
                        validation_data=(x_val_train, y_val_train),
                        verbose=1, epochs=35, steps_per_epoch=len(x_train), callbacks=callbacks_list)



if __name__ == '__main__':
    #freeze_support() # Optional under circumstances described in docs
    main()
