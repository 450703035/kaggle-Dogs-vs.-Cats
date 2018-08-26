# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:59:11 2018

@author: Administrator
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.nasnet import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
import numpy as np
import os
import h5py
import pandas as pd

train_data_dir = 'data/train_all/'
validation_data_dir = 'data/valid/'
test_dir = 'data/test/'

nasnet = "model/gap_nasnet.h5"
#bottleneck_train = 'model/bottleneck_features_train.npy'
#bottleneck_valid = 'model/bottleneck_features_validation.npy'
model_save_path = 'model/all_model.h5'

batch_size = 16
epochs = 50

filename = ["gap_Xception.h5", "gap_InceptionV3.h5"]
model_list = ['ResNet50', 'InceptionV3', 'Xception', 'VGG16', 'DenseNet169', 'Nasnet']

#nb_train_samples = len(os.listdir(train_data_dir+'train'))

nb_train_cats = len(os.listdir(train_data_dir+'dogs'))
nb_train_dogs = len(os.listdir(train_data_dir+'cats'))
nb_train_samples = nb_train_cats + nb_train_dogs

nb_valid_cats = len(os.listdir(validation_data_dir+'dogs'))
nb_valid_dogs = len(os.listdir(validation_data_dir+'cats'))
nb_validation_samples = nb_valid_cats + nb_valid_dogs

nb_test = len(os.listdir(test_dir + 'test'))



def model_init(model_type):
    if model_type == 'Xception':
        img_width, img_height = 299, 299
        from keras.applications.xception import preprocess_input
        model = applications.xception.Xception(weights='imagenet',
                                               include_top=False,
                                               input_shape=(img_width,img_height,3),
                                               pooling = 'avg',
                                               classes = 2)
        
    elif model_type == 'VGG16':
        img_width, img_height = 224, 224
        from keras.applications.vgg16 import preprocess_input
        model = applications.VGG16(weights='imagenet', 
                                    include_top=False,
                                    input_shape=(img_width,img_height,3),
                                    pooling = 'avg')
        
    elif model_type == 'ResNet50':
        img_width, img_height = 224, 224
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        model = applications.ResNet50(weights='imagenet', 
                                      include_top=False,
                                      pooling = 'avg',
                                      input_shape=(img_width,img_height,3))
        
    elif model_type == 'InceptionV3':
        img_width, img_height = 299, 299
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        model = applications.InceptionV3(weights='imagenet', 
                                         include_top=False,
                                         pooling = 'avg',
                                         input_shape=(img_width,img_height,3))
                    
    elif model_type == 'DenseNet169':
        img_width, img_height = 224, 224
        from keras.applications.densenet import preprocess_input, decode_predictions
        model = applications.DenseNet169(weights='imagenet', 
                                         pooling = 'avg',
                                         include_top=False,
                                         input_shape=(img_width,img_height,3))
        
    elif model_type == 'Nasnet':
        img_width, img_height = 224, 224
        from keras.applications.nasnet import preprocess_input, decode_predictions
        model = applications.nasnet.NASNetMobile (weights='imagenet', 
                                                  include_top=False, 
                                                  input_shape=(img_width,img_height,3),
                                                  pooling = 'avg',
                                                  )

    return model, preprocess_input, img_width, img_height



def save_bottlebeck_features(model, preprocess_input, img_width, img_height, num):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)
    
    bottleneck_features_train = model.predict_generator(train_generator)
    
    valid_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)
    
    bottleneck_features_valid = model.predict_generator(
        valid_generator, np.ceil(nb_validation_samples/batch_size))
    
    
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)
    
    bottleneck_features_test = model.predict_generator(
        test_generator, np.ceil(nb_test/batch_size))
    
    with h5py.File("model/gap_%s.h5"%model_list[num]) as h:
        h.create_dataset("train", data=bottleneck_features_train)
        h.create_dataset("t_label", data=train_generator.classes)
        h.create_dataset("valid", data=bottleneck_features_valid)
        h.create_dataset("v_label", data=valid_generator.classes)
        h.create_dataset("test", data=bottleneck_features_test)


def train_top_model():
    X_train = []
    X_valid = []
    X_test = []
    
    with h5py.File(nasnet, 'r') as h:
        #X_train.append(np.array(h['train']))
        X_train = np.array(h['train'])
        X_valid = np.array(h['valid'])
        X_test = np.array(h['test'])
        y_train = np.array(h['t_label'])
        y_valid = np.array(h['v_label'])
        
    #X_train = np.concatenate(X_train, axis=1)
    #X_test = np.concatenate(X_test, axis=1)
    
    X_train, y_train = shuffle(X_train, y_train)
    
    input_tensor = Input(X_train.shape[1:])
    x = Dropout(0.5)(input_tensor)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, x)
    
    model.compile(optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='min')
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks = [early_stop, checkpoint],
              validation_data=(X_valid, y_valid)
              )
    #model.save_weights(model_save_path) 


#def train_topall_model():
X_train = []
X_test = []
X_valid = []

for filename in ["model/gap_Nasnet.h5", "model/gap_Xception.h5", "model/gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_valid.append(np.array(h['valid']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['t_label'])
        y_valid = np.array(h['v_label'])

X_train = np.concatenate(X_train, axis=1)
X_valid = np.concatenate(X_valid, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.25)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='min')
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          callbacks = [early_stop, checkpoint],
          validation_data=(X_valid, y_valid)
          )


model = load_model('model/all_model.h5')
y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

df = pd.DataFrame()

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory(test_dir, (224, 224), shuffle=False, 
                                         batch_size=16, class_mode=None)
file_list = test_generator.filenames

for i in range(len(file_list)):
    index = file_list.index('test\\'+ str(i+1) +'.jpg')
    df.set_value(i+1, 'label', y_pred[index])

df.index.name = 'id'
df.to_csv('submit/pred_merge.csv')
df.head(10)

n = 5
#model, preprocess_input, img_width, img_height = model_init(model_list[n])    
#save_bottlebeck_features(model, preprocess_input, img_width, img_height, n)
#train_top_model()
#train_topall_model()

