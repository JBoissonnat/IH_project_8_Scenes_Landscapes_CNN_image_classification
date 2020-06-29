# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:35:34 2020

@author: jeanb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

print(tf.__version__)

def get_X_y(path, classes=False):
    os.chdir(path)
    dct = {path+"\\"+i+"\\"+j:i for i in os.listdir() for j in os.listdir(i) if j.endswith("jpg")}
    
    X_img = list(dct.keys())
    arrays = [img_to_array(load_img(X_img[i])) for i in range(len(X_img))]
    lst_err_shape = [i for i in range(len(arrays)) if arrays[i].shape != (150, 150, 3)]
    arrays = [i for j, i in enumerate(arrays) if j not in lst_err_shape]
    X = np.stack(arrays, axis=0)/255
    
    y_labels = list(dct.values())
    unique_classes = list(set(y_labels))
    unique_classes.sort()
    dct_labels = dict(zip(unique_classes, [i for i in range(len(set(y_labels)))]))
    y = list(map(dct_labels.get, y_labels))
    y = np.array([i for j, i in enumerate(y) if j not in lst_err_shape])
    
    if classes==True:
        return X, y, unique_classes
    else:
        return X, y

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                100*np.max(predictions_array),
                                classes[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(6))
  plt.yticks([])
  thisplot = plt.bar(range(6), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def eval_model(model):
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_acc)
    
    return None

def check_random_test_image(predictions):
    
    i = np.random.randint(0, X_test.shape[0], 1)[0]
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], y_test, X_test)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  y_test)
    plt.show()
    
    return None

def plot_accuracies_curves(history):
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return None

    plt.figure(figsize=(10,7))
    plt.plot(history_4_wit.history['acc'], "--")
    plt.plot(history_4_wit.history['val_acc'], color="limegreen")
    plt.title('Model 4 accuracy comparison per epoch (with images transformed', fontsize=15)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend(['Train accuracy', 'Test accuracy'], loc='upper left', fontsize=14)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(0.25, 0.90, step=0.05))
    plt.show()

### Import train and test data

X_test, y_test, classes = get_X_y(r"C:\data\intel-image-classification\seg_test\seg_test", classes=True)

X_train_full, y_train_full = get_X_y(r"C:\data\intel-image-classification\seg_train\seg_train")

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y_train_full)

X_test.shape
y_test.shape

X_train.shape
y_train.shape

X_valid.shape
y_valid.shape

from collections import Counter

list(Counter(y_valid).values())

classes

df = pd.DataFrame()
df["Train"] = list(Counter(y_train).values())
df["Valid"] = list(Counter(y_valid).values())
df["Test"] = list(Counter(y_test).values())
df["Classes"] = classes
df=df.set_index('Classes')

plt.figure(figsize=(20,10))
df.plot.barh(figsize=(10,7), title="Number of images per classes")

plt.rcParams["axes.grid"] = False

i = np.random.randint(0, X_test.shape[0], 1)[0]
plt.imshow(X_test[i])
plt.title(classes[y_test[i]])

################# MODEL 1

model_1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(150, 150, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6)
])

model_1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
history_1=model_1.fit(X_train, y_train, epochs=10,
          verbose=1, validation_data=(X_valid, y_valid))

################# MODEL 4

### Train the model 4



model_4 = keras.Sequential([
    
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6)
])

model_4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
history_4=model_4.fit(X_train, y_train, epochs=10,
          verbose=1, validation_data=(X_valid, y_valid), batch_size=1100) # Shuffle ?

### Test the model
#os.chdir("C:\data\models")
#model = tf.keras.models.load_model('model_4_h.h5')

eval_model(model_4)

probability_model = tf.keras.Sequential([model_4, 
                                         tf.keras.layers.Softmax()])

predictions_4 = probability_model.predict(X_test)

### Check prediction for images

check_random_test_image(predictions_4)

#### Plot accuracies curves

### model saving
#os.chdir("C:\data\models")
#model.save("model_4.h5")

#from collections import Counter

#list(Counter(y_valid).values())

plot_accuracies_curves(history_4)

y_pred = [np.argmax(predictions_4[i]) for i in range(len(predictions_4))]

conf_mat = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(12,10))
plt.title("Confusion Matrix for Model 4 CNN predictions")
sns.heatmap(conf_mat,annot=True,annot_kws={'fontsize':13},cmap='viridis',fmt='g',cbar=False,
            xticklabels=classes, yticklabels=classes)


################# MODEL 4 with IT

### Transforming images

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

## train transformed images
train_iterator = datagen.flow(X_train, y_train, batch_size=4000)
X_train_ti, y_train_ti = train_iterator.next()

## valid transformed images
valid_iterator = datagen.flow(X_valid, y_valid, batch_size=1000)
X_valid_ti, y_valid_ti = valid_iterator.next()

# X and y for train wit
X_train_wit_lst = list(X_train) + list(X_train_ti)

X_train_wit = np.stack(X_train_wit_lst, axis=0)

y_train_wit = np.array(list(y_train) + list(y_train_ti))

# X and y for valid wit

X_valid_wit_lst = list(X_valid) + list(X_valid_ti)

X_valid_wit = np.stack(X_valid_wit_lst, axis=0)

y_valid_wit = np.array(list(y_valid) + list(y_valid_ti))


i = np.random.randint(2798, X_valid_wit.shape[0], 1)[0]
plt.imshow(X_valid_wit[i])
plt.title(classes[y_valid_wit[i]])

df = pd.DataFrame()
df["Train"] = list(Counter(y_train_wit).values())
df["Valid"] = list(Counter(y_valid_wit).values())
df["Test"] = list(Counter(y_test).values())
df["Classes"] = classes
df=df.set_index('Classes')

plt.figure(figsize=(20,10))
df.plot.barh(figsize=(10,7), title="Number of images per classes (with transformed images")

### Train the model 4 wit

model_4_wit = keras.Sequential([
    
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6)
])

model_4_wit.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
history_4_wit = model_4_wit.fit(X_train_wit, y_train_wit, epochs=10,
          verbose=1, validation_data=(X_valid_wit, y_valid_wit), batch_size=1500) # Shuffle ?

eval_model(model_4_wit)

probability_model = tf.keras.Sequential([model_4_wit, 
                                         tf.keras.layers.Softmax()])

predictions_4_wit = probability_model.predict(X_test)

check_random_test_image(predictions_4_wit)

plot_accuracies_curves(history_4_wit)




