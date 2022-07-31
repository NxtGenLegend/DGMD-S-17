import os 
import cv2
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

resultsdic = {}
cnnreslist = []

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen=ImageDataGenerator(rescale=1./255)

input_shape=(224,224,3)

test_generator = test_datagen.flow_from_directory(test_dir,shuffle=True,target_size=(224,224),batch_size=32)

for training_name in train_labels:
    train_dir = train_path + "/" + training_name
    train_generator = train_datagen.flow_from_directory(train_dir,target_size=(224,224),batch_size=32)

model = Sequential()
model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu',name="conv2d_1"))
model.add(MaxPooling2D(pool_size=(3, 3),name="max_pooling2d_1"))
model.add(Conv2D(32, (3, 3),activation='relu',name="conv2d_2"))
model.add(MaxPooling2D(pool_size=(2, 2),name="max_pooling2d_2"))
model.add(Conv2D(64, (3, 3),activation='relu',name="conv2d_3"))
model.add(MaxPooling2D(pool_size=(2, 2),name="max_pooling2d_3"))   
model.add(Flatten(name="flatten_1"))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))          
model.add(Dense(num_classes,activation='softmax'))
model.summary()

validation_generator = train_datagen.flow_from_directory(
                       test_dir,
                       target_size=(224, 224),
                       batch_size=32)

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
history1 = model.fit(
    train_generator,#egitim verileri
    steps_per_epoch=None,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=None,
    verbose=1,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)],
    shuffle=True
    )

#model.save('plant_disease_Cnn.h5')
model.save('plant_disease_Cnn.h5')

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
#model_cnn=load_model('plant_disease_Cnn.h5')

model_cnn=load_model('plant_disease_Cnn.h5')

classes=list(train_generator.class_indices.keys())
# Pre-Processing test data same as train data.
def prepare(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

#take testsetsglobal from the models code in order to test

for filename in os.scandir(testsetsglobal):
    img_url = filename.path
    #img_url='/content/drive/MyDrive/Colab Notebooks/dataset/plant__leaf/val/Apple__Healthy/78e648c6-a360-4fa8-b8ab-1225b164b7fd___RS_HL 7243.JPG'
    result_cnn = model_cnn.predict([prepare(img_url)])
    disease=image.load_img(img_url)
    plt.imshow(disease)
    classresult=np.argmax(result_cnn,axis=1)
    cnnreslist.append(classes[classresult[0]])
    resultsdic["cnn"] = cnnreslist
    print(classes[classresult[0]])