# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 23:18:46 2018

@author: Administrator
"""


from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

model_path = 'model/xception_model_0.01367.h5'
test_data_dir = 'data/test/'
model = load_model(model_path)


submiss_file = 'submit/sample_submission_0.01367.csv'

df = pd.DataFrame()


img_width, img_height = 299, 299
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(test_data_dir, 
                                    target_size=(img_width, img_height),
                                    shuffle=False,
                                    class_mode="binary",
                                    batch_size=16)


test_pred = model.predict_generator(test_generator, verbose=1)
test_pred = test_pred.clip(min=0.005, max=0.995)

file_list = test_generator.filenames

for i in range(len(file_list)):
    index = file_list.index('test\\'+ str(i+1) +'.jpg')
    df.set_value(i+1, 'label', test_pred[index])

'''
for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('\\')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', test_pred[i])
'''

df.index.name = 'id'
df.to_csv(submiss_file)
df.head(10)