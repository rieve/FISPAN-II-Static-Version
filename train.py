#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.layers import Dropout
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import h5py



# In[18]:



# In[3]:


# Define the percentage of data for each split
train_split = 0.7
test_split = 0.2
val_split = 0.1


# In[43]:


data_path = 'D:/Sign language/Final/Image'
train_path = 'D:/Sign language/Final/train'
test_path = 'D:/Sign language/Final/test'
val_path = 'D:/Sign language/Final/valid'


# In[44]:


# Create train, validation, and test directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


# In[45]:


# Loop over each category and copy images to train, validation, and test folders
for category in os.listdir(data_path):
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)
    
    image_files = os.listdir(os.path.join(data_path, category))
    num_images = len(image_files)
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)
    num_test = num_images - num_train - num_val
    
    # Shuffle image files
    random.shuffle(image_files)
    
    # Copy the first num_train images to the train directory
    for i in range(num_train):
        src = os.path.join(data_path, category, image_files[i])
        dst = os.path.join(train_path, category, image_files[i])
        shutil.copyfile(src, dst)
    
    # Copy the next num_val images to the validation directory
    for i in range(num_train, num_train+num_val):
        src = os.path.join(data_path, category, image_files[i])
        dst = os.path.join(val_path, category, image_files[i])
        shutil.copyfile(src, dst)
    
    # Copy the remaining images to the test directory
    for i in range(num_train+num_val, num_images):
        src = os.path.join(data_path, category, image_files[i])
        dst = os.path.join(test_path, category, image_files[i])
        shutil.copyfile(src, dst)


# In[4]:


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                               rotation_range=5,
                                                               width_shift_range=0.2,
                                                               height_shift_range=0.2,
                                                               shear_range=0.2,
                                                               zoom_range=0.2,
                                                               horizontal_flip=True)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory('D:/Sign language/Final/train',
                                                  target_size=(224, 224),
                                                  batch_size=15,   #add 26 for a to z in all places where there is 15
                                                  class_mode='categorical')

val_dataset = val_datagen.flow_from_directory('D:/Sign language/Final/valid',
                                              target_size=(224, 224),
                                              batch_size=15,
                                              class_mode='categorical')

test_dataset = test_datagen.flow_from_directory('D:/Sign language/Final/test',
                                                target_size=(224, 224),
                                                batch_size=15,
                                                class_mode='categorical')


# In[5]:


# # Load the train dataset
# import cv2
# import matplotlib.pyplot as plt
# # first_image = np.array(test_dataset[0][0][0])
# first_image=cv2.imread('0.png')
# # Print the shape of the first image
# print(first_image.shape)
# # gray = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
# # print(gray.shape)
# # plt.imshow(first_image)
# # plt.show()


# In[6]:


model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224,3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(15, activation='softmax') #26
])


# In[7]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[8]:


history = model.fit(train_dataset, 
                    epochs=12, 
                    validation_data=val_dataset)


# In[8]:


test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)


# In[4]:


# Plot the training and validation accuracy and loss over time
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[12]:


get_ipython().system('pip install keras --upgrade')


# In[15]:


# Save the model for future use
model.save('sign_language_model.h5')


# In[ ]:




