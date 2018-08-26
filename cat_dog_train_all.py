# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:10:33 2018

@author: Administrator
"""

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import os
import numpy as np

# path to the model weights files.
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.

train_data_dir = 'data/train_all/'
validation_data_dir = 'data/valid/'
#train_data_dir = 'data/train/'
#validation_data_dir = 'data/validation/'
model_save_path = 'model/xception_model.h5'

nb_train_samples = (len(os.listdir(train_data_dir+'dogs')) +
                        len(os.listdir(train_data_dir+'cats')))
nb_validation_samples = (len(os.listdir(validation_data_dir+'dogs')) +
                        len(os.listdir(validation_data_dir+'cats')))
epochs = 50
batch_size = 16


# save the anomaly filename list
model_list = ['ResNet50', 'InceptionV3', 'Xception', 'VGG16', 'DenseNet169']

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
        from keras.applications.vgg16 import preprocess_input, decode_predictions
        model = applications.VGG16(weights='imagenet', 
                            include_top=False, input_shape=(img_width,img_height,3))
        
    elif model_type == 'ResNet50':
        img_width, img_height = 224, 224
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        model = applications.ResNet50(weights='imagenet', 
                            include_top=False, input_shape=(img_width,img_height,3))
        
    elif model_type == 'InceptionV3':
        img_width, img_height = 299, 299
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        model = applications.InceptionV3(weights='imagenet', 
                            include_top=False, input_shape=(img_width,img_height,3))
        
    elif model_type == 'DenseNet169':
        img_width, img_height = 224, 224
        from keras.applications.densenet import preprocess_input, decode_predictions
        model = applications.DenseNet169(weights='imagenet', 
                                include_top=False, input_shape=(img_width,img_height,3))
        
        
    return model, preprocess_input, img_width, img_height

#7000 valid        
#0.5  256  xception_model_0.025.h5
#0.3  256  xception_model0.02338.h5

#5000 valid
#0.3  256  LeakyReLU  adam  
#0.3  256  PReLU  adam    0.01367
    
dropout = 0.3

model, preprocess_input, img_width, img_height = model_init(model_list[2])

top_model = Sequential()
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
#top_model.add(GlobalAveragePooling2D())
top_model.add(Dropout(dropout))
#top_model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#top_model.add(BatchNormalization())
#top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(256, activation='linear'))
#top_model.add(LeakyReLUalpha=.001))
top_model.add(PReLU())
top_model.add(Dropout(dropout))
top_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=model.input, outputs=top_model(model.output))

#for i, layer in enumerate(model.layers):
#   print(i, layer.name)

for layer in model.layers[:130]:
    layer.trainable = False


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.

#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
#                      epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy',
              #optimizer='adadelta',
              optimizer='adam',
              metrics=['accuracy'])

#model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#train_datagen = ImageDataGenerator(rescale=1. / 255)

#valid_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

nb_train_steps = np.ceil(nb_train_samples/batch_size)
nb_validation_steps = np.ceil(nb_validation_samples/batch_size)

early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='min')
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=False, mode='auto', period=1)

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_steps,
    callbacks = [checkpoint, early_stop],
    verbose=1)
