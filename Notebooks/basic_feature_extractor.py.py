'''
Basic feature extractor version gives accuracy around 84% for training set but low performance on the validation set (around 22%),
which means it has been over fitted.
this model, in the first layer, uses some activation functions other that ReLU such as Cosine and Sin activation fc.
I created three parallel layers that take the input of the network and apply one of the ReLU, Cosine, and Sin activation
functions, and then, I concatenate them to have a wide layer, and then added other layers to the network. The dropout
layer is also been added to prevent over fitting.

'''
# Run if using tensorflow2.0+
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Flatten
import matplotlib.pyplot as plt

dataset = pd.read_csv('../Data/train.csv')
train_size = int(dataset.shape[0] * 0.75)
df_train = dataset[:train_size]
df_test = dataset[train_size:]


def create_model():
    input1 = tf.keras.layers.Input(shape=(1,))
    x1 = tf.keras.layers.Dense(20, activation=tf.keras.backend.sin)(input1)
    x2 = tf.keras.layers.Dense(20, activation=tf.keras.backend.cos)(input1)
    x3 = tf.keras.layers.Dense(20, activation=tf.keras.backend.relu)(input1)
    added = tf.keras.layers.concatenate([x1, x2, x3])
    l0 = tf.keras.layers.Dropout(0.3)(added)
    l1 = tf.keras.layers.Dense(200, activation='relu')(l0)
    l2 = tf.keras.layers.Dropout(0.3)(l1)
    l3 = tf.keras.layers.Dense(200, activation='relu')(l2)
    out = tf.keras.layers.Dense(11, activation='softmax')(l3)
    model = tf.keras.models.Model(inputs=[input1], outputs=out)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss = tf.keras.losses.BinaryCrossentropy()
    # loss = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam()
    # opt = tf.keras.optimizers.SGD
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    return model


x = df_train.signal
y = df_train.open_channels
model = create_model()
# model.summary()
model.fit(x, y, epochs=1, validation_split=.3, verbose=1)

print('Done')
predictions = model.predict(x)

df_test = df_train.sample(frac=0.2)
x_test = df_test.signal
y_test = df_test.open_channels
prediction = model.evaluate(x_test, y_test)

print('Test Accuracy: %.3f' % prediction[1])
