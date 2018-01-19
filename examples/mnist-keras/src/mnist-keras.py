from __future__ import print_function
import os
import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from kogu import Kogu

numpy.random.seed(555)

epochs = 10
batch_size = 128
dropout = 0.2
fc1 = 512
fc2 = 512
learning_rate = 0.001

Kogu.load_parameters()

Kogu.update_parameters({
    "epochs": epochs,
    "batch_size": batch_size,
    "dropout": dropout,
    "fc1": fc1,
    "fc2": fc2,
    "learning_rate": learning_rate,
}, output=True)

Kogu.plot(plot_type="line", y_label="train, validation", series=["train", "validation"], name="Accuracy")

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(fc1, activation='relu', input_shape=(784,)))
model.add(Dropout(dropout))
model.add(Dense(fc2, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

class SendAcc(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        Kogu.metrics({
            "train": logs.get('acc'),
            "validation": logs.get('val_acc'),
        }, iteration=epoch)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test, y_test),
          callbacks=[SendAcc()])

score = model.evaluate(x_test, y_test, verbose=0)
Kogu.metrics({"accuracy": score[1]})
