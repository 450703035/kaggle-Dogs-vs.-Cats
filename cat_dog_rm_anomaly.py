# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:35:12 2018

@author: dawa
"""

import os
import random
import shutil

anomaly_dir = ''
train_dog_dir = 'data\\train_all\\dogs'
train_cat_dir = 'data\\train_all\\cats'
anomaly_dir = 'data\\Anomaly_data\\Obvious'

valid_dog_dir = 'data\\valid\\dogs'
valid_cat_dir = 'data\\valid\\cats'


anomaly_list = os.listdir(anomaly_dir)

for i in range(len(anomaly_list)):
    dog_file = os.path.join(train_dog_dir, anomaly_list[i])
    cat_file = os.path.join(train_cat_dir, anomaly_list[i])
    if (os.path.exists(dog_file)):
        print(dog_file)
        os.remove(dog_file)
        
    if (os.path.exists(cat_file)):
        print(cat_file)
        os.remove(cat_file)
        
#valid_dog = []
#valid_cat = []
#test_dog = []

dog_list = os.listdir(train_dog_dir)
cat_list = os.listdir(train_cat_dir)

valid_dog = random.sample(dog_list, 2500) 
valid_cat = random.sample(cat_list, 2500) 


for name in valid_dog:
    srcname = os.path.join(train_dog_dir, name)
    dstname = os.path.join(valid_dog_dir, name)
    shutil.move(srcname,dstname)

for name in valid_cat:
    srcname = os.path.join(train_cat_dir, name)
    dstname = os.path.join(valid_cat_dir, name)
    shutil.move(srcname,dstname)

#shutil.move(srcfile,dstfile)          #移动文件