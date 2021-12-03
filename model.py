#!/usr/bin/env python
# coding: utf-8

# In[2]:

import local_variables
from keras.layers.normalization.batch_normalization import BatchNormalization
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import keras

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

from matplotlib import pyplot
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
data_gen = ImageDataGenerator(rescale=1.0/255)



# In[3]:


#tf.test.is_built_with_cuda()
tf.device('CPU')


# In[4]:


# In[5]:




# In[6]:


path = local_variables.pictures_edits_path


# In[7]:


img_x = 64
img_y = 128


# In[8]:


def gen_training(img_y, img_x):
    batch_size = 500 #32

    train_generator = data_gen.flow_from_directory(
        path + r'\train',
        target_size=(img_y, img_x),
        batch_size=batch_size,
        classes=['no', 'one'],
        seed=12345,
        shuffle=True)

    return train_generator


# In[9]:


def gen_validation(img_y, img_x):
    batch_size = 500 #32

    test_generator = data_gen.flow_from_directory(
        path + r'\test',
        target_size=(img_y, img_x),
        batch_size=batch_size,
        classes=['no', 'one'],
        seed=12345,
        shuffle=True)
    return test_generator


# In[10]:


def make_convnet(X, Y, x_test, y_test, img_y, img_x):
    num_classes = 2

    print(X[0].shape)
    model_bi = Sequential()
    
    model_bi.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1),
                        activation='relu',
                        input_shape=(img_y, img_x, 3)))
    model_bi.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model_bi.add(BatchNormalization())
    model_bi.add(Conv2D(64, (5, 5), activation='relu'))
    model_bi.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model_bi.add(BatchNormalization())
    model_bi.add(Flatten())
    model_bi.add(Dense(64, activation='relu'))

    model_bi.add(Dense(num_classes, activation='sigmoid'))
    

    '''
    model_bi.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu',
                        input_shape=(img_y, img_x, 3)))
    model_bi.add(keras.layers.Flatten(input_shape=(img_y, img_x)))
    model_bi.add(keras.layers.Dense(32, activation='relu'))
    model_bi.add(keras.layers.Dense(2,activation='sigmoid'))
    '''

    print("here")
    #loss='SparseCategoricalCrossentropy'
    model_bi.compile(loss = 'categorical_crossentropy',
                     optimizer='sgd',
                     metrics=['accuracy'])

    print("here1.2")
    history = model_bi.fit(X, Y,
                           batch_size= 50,#32,
                           epochs=20,
                           verbose=1,
                           callbacks=[early_stopping],
                           validation_data=(x_test, y_test))
    print("here2")
    _, Train_score_bi = model_bi.evaluate(X, Y, verbose=1)
    print("here3")
    _, Validation_score_bi = model_bi.evaluate(x_test, y_test, verbose=0)
    print('Train score: %.3f, Validation score: %.3f' % (Train_score_bi, Validation_score_bi))

    #print('history.history: ' + str(history.history['loss'][4]))
    #print('history.history: ' + str(history.history['loss'][5]))
    #print('history.history: ' + str(history.history['loss'][6]))
    #print(history.history)

    # plot loss during training
    pyplot.figure(figsize=(15, 10))
    pyplot.subplot(221)
    pyplot.title('Loss')
    pyplot.ylim(0,10)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()

    # plot accuracy during training
    pyplot.subplot(222)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='validation')
    pyplot.legend()

    pyplot.savefig('Last_Figure')
    pyplot.show()
    return 0


# In[11]:


train_generator = gen_training(img_y, img_x)
test_generator = gen_validation(img_y, img_x)


# In[12]:


Xbatch, Ybatch = train_generator.next()
x_test, y_test = test_generator.next()


# In[13]:


#pyplot.imshow(x_test[4]);
#pyplot.imshow(Xbatch[8])
pyplot.show()


# In[ ]:


a_model_2 = make_convnet(Xbatch, Ybatch, x_test, y_test, img_y, img_x)


# In[ ]:


#a_model_3 = make_convnet(Xbatch, Ybatch, x_test, y_test, 64) ## added layers


# In[ ]:





# In[ ]:


