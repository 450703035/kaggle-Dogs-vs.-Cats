# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:12:14 2018

@author: dawa
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
#from keras import optimizers
#from keras.models import Sequential
#from keras.models import Model
#from keras.layers import Dropout, Flatten, Dense
#from keras.preprocessing import image
import numpy as np
import json
import shutil  
import pandas as pd
import os

dog = [
'n02085620','n02085782','n02085936','n02086079','n02086240',
'n02086646','n02086910','n02087046','n02087394','n02088094',
'n02088238','n02088364','n02088466','n02088632','n02089078',
'n02089867','n02089973','n02090379','n02090622','n02090721',
'n02091032','n02091134','n02091244','n02091467','n02091635',
'n02091831','n02092002','n02092339','n02093256','n02093428',
'n02093647','n02093754','n02093859','n02093991','n02094114',
'n02094258','n02094433','n02095314','n02095570','n02095889',
'n02096051','n02096177','n02096294','n02096437','n02096585',
'n02097047','n02097130','n02097209','n02097298','n02097474',
'n02097658','n02098105','n02098286','n02098413','n02099267',
'n02099429','n02099601','n02099712','n02099849','n02100236',
'n02100583','n02100735','n02100877','n02101006','n02101388',
'n02101556','n02102040','n02102177','n02102318','n02102480',
'n02102973','n02104029','n02104365','n02105056','n02105162',
'n02105251','n02105412','n02105505','n02105641','n02105855',
'n02106030','n02106166','n02106382','n02106550','n02106662',
'n02107142','n02107312','n02107574','n02107683','n02107908',
'n02108000','n02108089','n02108422','n02108551','n02108915',
'n02109047','n02109525','n02109961','n02110063','n02110185',
'n02110341','n02110627','n02110806','n02110958','n02111129',
'n02111277','n02111500','n02111889','n02112018','n02112137',
'n02112350','n02112706','n02113023','n02113186','n02113624',
'n02113712','n02113799','n02113978']


wolf = [
'n02114367','n02114548','n02114712','n02114855',
'n02115641','n02115913','n02116738','n02117135']

cat = [
'n02123045','n02123159','n02123394','n02123597',
'n02124075','n02125311','n02127052']

fox = [
'n02119022','n02119789',
'n02120079','n02120505']

panther = [
'n02128385','n02128757',
'n02128925']

cat_dog = panther + fox + cat + wolf + dog

json_file = 'imagenet_class_index.json'
train_data_dir = 'data/train_all/'
anomaly_data_dir = 'data/anomaly_all/'
#train_data_dir = 'data/anomaly0'


# save the anomaly filename list
model_list = ['ResNet50', 'InceptionV3', 'Xception', 'VGG16', 'DenseNet169']
top_list = [15, 10, 15, 20, 10] 
error_list = []

num  = 4
top = top_list[num]
model_name = model_list[num]

def model_init(model_type):
    if model_type == 'Xception':
        from keras.applications.xception import preprocess_input, decode_predictions
        model = applications.xception.Xception(weights='imagenet', include_top=True)
        img_width, img_height = 299, 299
    elif model_type == 'VGG16':
        from keras.applications.vgg16 import preprocess_input, decode_predictions
        model = applications.VGG16(weights='imagenet', include_top=True)
        img_width, img_height = 224, 224
    elif model_type == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        model = applications.ResNet50(weights='imagenet', include_top=True)
        img_width, img_height = 224, 224
    elif model_type == 'InceptionV3':
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        model = applications.InceptionV3(weights='imagenet', include_top=True)
        img_width, img_height = 299, 299
    elif model_type == 'DenseNet169':
        from keras.applications.densenet import preprocess_input, decode_predictions
        model = applications.DenseNet169(weights='imagenet', include_top=True)
        img_width, img_height = 224, 224
        
    return model, preprocess_input, decode_predictions, img_width, img_height


model,preprocess_input,decode_predictions,img_width,img_height = model_init(model_name)
print('Model loaded.')
#model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=50,
    class_mode='categorical',
    shuffle=False)

# predict
result = model.predict_generator(train_generator, verbose=True)
pred = decode_predictions(result, top)

# 打印预测信息
#print(result_cat.shape)
#print('Predicted:', pred)

# parse json file
file = open(json_file,'r',encoding='utf-8')  
imagenet_c = json.load(file)
imagenet_c_l = list(imagenet_c.values())
imagenet_c_n = np.array(imagenet_c_l)

anomaly_file = 'anomaly_file.csv'

if (os.path.exists(anomaly_file)):
    df = pd.read_csv(anomaly_file)
else:
    #df = pd.DataFrame(columns=model_list)
    df = pd.DataFrame()
    df.to_csv(anomaly_file)
    df = pd.read_csv(anomaly_file)
index = range(1,1001)

# 预测异常分析
anomaly_ok = 0
anomaly_num = 0
for i in range(len(pred)):
    for j in range(top):
        if pred[i][j][0] in cat_dog:
            '''
            name = train_generator.filenames[i]
            if (name.find('error') != -1):
                anomaly_ok+=1
                print("error : %s" % train_generator.filenames[i])
            '''
            break;
    else:
        anomaly_num+=1
        
        # 1000分类对于的类别名
        pred_name = imagenet_c_n[np.argwhere(imagenet_c_n==
                   pred[i][0][0])[0][0]][1]
        
        name = train_generator.filenames[i]
        error_list.append(name)
       
        #df = df.append({model_name:name}, ignore_index=True)
        #异常文件名
        filename = name[name.find('\\')+1:]
        #复制异常文件到指定文件夹
        shutil.copy(train_data_dir+name, anomaly_data_dir+filename)
        
        print("ERROR>> %18s    num:%6d     tag:%s    name:%18s   prob:%s" 
              %(train_generator.filenames[i], i, pred[i][0][0],
                pred_name, pred[i][0][2]))
        
print("%d个样本预测错%d  anomaly_ok:%d" %(len(pred), anomaly_num, anomaly_ok))
#df = pd.DataFrame(data=error_list, columns=model_list) 
df[model_name] = pd.Series(error_list)
df.to_csv(anomaly_file)