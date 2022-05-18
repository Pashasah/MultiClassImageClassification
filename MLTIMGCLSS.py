# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:50:40 2022

@author: sp7012
"""

# Importing modules 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
WidthI=100;

train_images = []       
train_labels = []
shape = (WidthI,WidthI)  
train_path = '../MltiClassiMG/archive/train_zip/train'

for filename in os.listdir('../MltiClassiMG/archive/train_zip/train'):
    if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(train_path,filename))
        
        # Spliting file names and storing the labels for image in list
            train_labels.append(filename.split('_')[0])
        
        # Resize all images to a specific shape
            img = cv2.resize(img,shape)
        
            train_images.append(img)

# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array
train_images = np.array(train_images)

# Splitting Training data into train and validation dataset
x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=1)

# Processing testing data
# -> appending images in a list 'test_images'
# -> appending labels in a list 'test_labels'
# The test data contains labels as well also we are appending it to a list but we are'nt going to use it while training.

test_images = []
test_labels = []
shape = (WidthI,WidthI)
test_path = '../MltiClassiMG/archive/test_zip/test'

for filename in os.listdir('../MltiClassiMG/archive/test_zip/test'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path,filename))
        
        # Spliting file names and storing the labels for image in list
        test_labels.append(filename.split('_')[0])
        
        # Resize all images to a specific shape
        img = cv2.resize(img,shape)
        
        test_images.append(img)
        
# Converting test_images to array
test_images = np.array(test_images)


# Visualizing Training data
print(train_labels[0])
plt.imshow(train_images[0])


# Visualizing Training data
print(train_labels[4])
plt.imshow(train_images[4])

# Creating a Sequential model
model= Sequential()
model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=(WidthI,WidthI,3,)))
model.add(Conv2D(filters=60,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))

model.add(Flatten())

model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))

model.add(Dense(4,activation = 'softmax'))
    
model.compile(
              loss='categorical_crossentropy', 
              metrics=['acc'],
              optimizer='adam'
             )

# Model Summary
model.summary()

history = model.fit(x_train,y_train,epochs=50,batch_size=50,validation_data=(x_val,y_val))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluating model on validation data
evaluate = model.evaluate(x_val,y_val)
print(evaluate)

# Testing predictions and the actual label
checkImage = test_images[0:1]
checklabel = test_labels[0:1]

predict = model.predict(np.array(checkImage))

output = { 0:'apple',1:'banana',2:'mixed',3:'orange'}

print("Actual :- ",checklabel)
print("Predicted :- ",output[np.argmax(predict)])