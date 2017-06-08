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
from subprocess import check_output
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
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


def roi(pathtrain):
    for typ in types:
        for img in os.listdir(pathtrain + '/' + typ):
            image = pathtrain + '/'+typ+'/' + img
            os.chdir(pathtrain + '/'+typ+'/')
            ii=cv2.imread(image)
            #cv.imshow('image',ii[:,:,1])
            #cv.waitKey(0)
            b,g,r = cv2.split(ii)
            rgb_img = cv2.merge([r,g,b])
            rgb_img1 = pc.rgb_to_hsv(rgb_img)
            indices = np.where(rgb_img1[:,:,0]<0.7)
            rgb_img1[:,:,0][indices]=0
            rgb_img1[:,:,1][indices]=0
            rgb_img1[:,:,2][indices]=0
            rgb_img1 = pc.hsv_to_rgb(rgb_img1).astype(np.uint8)
            pp.imsave(fname = img.split('.')[0] + '_trans.jpg',arr = rgb_img1)
    return fname

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
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
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
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 32, 32))) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def main():
    os.chdir('D:\Facultad\Master\Segundo cuatrimestre\SIGE\PracticaFinal')
    test = glob.glob("pruebaTest/*.jpg")
    test = pd.DataFrame([[p[11:len(p)],p] for p in test], columns = ['image','path'])
    train= glob.glob("prueba1/**/*.png")+glob.glob("prueba2/**/*.jpg")
    train = pd.DataFrame([[p[0][8:14],p[0][15:len(p)],p] for p in train], columns = ['type','image','path'])

    train = im_stats(train)
    train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
    train_data = normalize_image_features(train['path'])

    np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)

    print(len(train))

    le = LabelEncoder()
    train_target = le.fit_transform(train['type'].values)
    print(le.classes_) #in case not 1 to 3 order
    np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)
    test_data = normalize_image_features(test['path'])

    np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)

    test_id = test.image.values
    np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)

    train_data = np.load('train.npy')
    train_target = np.load('train_target.npy')

    x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)


    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
    datagen.fit(train_data)


    model = create_model()
    print(x_train.shape)
    print(y_train.shape)
    model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=35, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))

    test_data = np.load('test.npy')
    test_id = np.load('test_id.npy')

    pred = model.predict_proba(test_data)
    df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
    df['image_name'] = test_id
    df.to_csv('submission0009.csv', index=False)

if __name__ == '__main__':
    #freeze_support() # Optional under circumstances described in docs
    main()
