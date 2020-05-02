'''
Best result was around 72% accuracy
An ensemble of the basic feature extractor. You can change the number of trained model by changing nb_model_per_class
parameter in the code.
After training a voting method has been used to aggregate the result of all the trained model, and class with highest
number of votes has been selected as the predicted class.
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


def filter_data(df, class_number, model_number):
    data = df.copy()
    if model_number == 1:
        data = data[model_number * 1000:]
    data.loc[data.open_channels != class_number, 'open_channels'] = 100
    data.loc[data.open_channels == class_number, 'open_channels'] = 1
    data.loc[data.open_channels == 100, 'open_channels'] = 0
    g = data.groupby('open_channels')
    balanced = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    return balanced


def create_model():
    input1 = tf.keras.layers.Input(shape=(1,))
    x1 = tf.keras.layers.Dense(100, activation=tf.keras.backend.sin)(input1)
    x2 = tf.keras.layers.Dense(100, activation=tf.keras.backend.cos)(input1)
    x3 = tf.keras.layers.Dense(100, activation=tf.keras.backend.relu)(input1)
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


def accuraccy(predictions, labels):
    accuraccy = np.mean(predictions == labels)
    return accuraccy


nb_model_per_class = 2
model_pool = {i: None for i in range(nb_model_per_class)}

for j in range(nb_model_per_class):
    data = df_train.sample(frac=0.01)
    x = data.signal
    y = data.open_channels
    model = create_model()
    model.fit(x, y, epochs=1, validation_split=.3, verbose=1)
    model_pool[j] = model

print('Done')
# predictions = model.predict(x)
nb_classes = 11
df_test = df_train.sample(frac=0.02)
x_test = df_test.signal
y_test = df_test.open_channels
ensemble_prediction = np.zeros((y_test.shape[0], nb_classes))
for j in range(nb_model_per_class):
    prediction = model_pool[j].predict(x_test)
    # p = np.argmax(prediction)
    ensemble_prediction += prediction

voting_prediction = ensemble_prediction.argmax(axis=1)
acc = accuraccy(voting_prediction, y_test)
print('Test Accuracy: %.3f' % acc)
