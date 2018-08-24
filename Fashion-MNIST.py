#Followed Medium tutorial:
#https://medium.com/deep-learning-turkey/deep-learning-lab-episode-1-fashion-mnist-c7af60029836

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import print_summary
from keras.optimizers import Adam
from keras.regularizers import l2
import os


###Pre-Processing###
batch_size = 32
num_classes = 100
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_fashion_mnist_trained_model.h5'
#Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#Reshape the data since images are Grayscale
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
input_shape = (28, 28, 1) #HWC Format
#Convert labels to categorical matrix structure
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
###Build the model###
model = Sequential ()
model.add (Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=input_shape))
model.add (Activation('relu'))
model.add (Conv2D(32, (5, 5), kernel_regularizer=l2(0.01)))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))
model.add (Dropout(0.25))

model.add (Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
model.add (Activation('relu'))
model.add (Conv2D(64, (5, 5), kernel_regularizer=l2(0.01)))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Flatten())
model.add (Dense(512))
model.add (Activation('relu'))
model.add (Dropout(0.5))
model.add(Dense(num_classes))
model.add (Activation('softmax'))

#Optimize weights during the backpropagation using Adam
opt = Adam (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#Normalize the images in the dataset (more preprocessing)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Compile the model
model.compile (loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit (x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

#Results
scores = model.evaluate (x_test, y_test, verbose=1)
print ('Test Loss: ', scores[0])
print ('Test Accuracy: ', scores[1])
