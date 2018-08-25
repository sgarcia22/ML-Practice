#Following tutorial from Medium: https://medium.com/deep-learning-turkey/deep-learning-lab-episode-2-cifar-10-631aea84f11e
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import print_summary, to_categorical
import sys
import os
###Pre-Processing###
batch_size = 64
num_classes = 10
epochs = 100
model_name = 'keras_cifar10_model'
save_dir = os.path.join(os.getcwd(), model_name)
#Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#Convert labels to categorical matrix structure
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
#Normalize the images in the dataset (more preprocessing)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
###Build the model###
model = Sequential ()
model.add (Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))
model.add (Dropout(0.3))

model.add (Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))
model.add (Dropout(0.3))

model.add (Conv2D(128, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))
model.add (Dropout(0.3))

model.add (Flatten())
model.add (Dense(80))
model.add (Activation('relu'))
model.add (Dropout(0.3))
model.add(Dense(num_classes))
model.add (Activation('softmax'))
#Optimize weights during the backpropagation using Adam
opt = SGD (lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
#Compile the model
model.compile (loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit (x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)
#Results
scores = model.evaluate (x_test, y_test, verbose=1)
print ('Test Loss: ', scores[0])
print ('Test Accuracy: ', scores[1])
