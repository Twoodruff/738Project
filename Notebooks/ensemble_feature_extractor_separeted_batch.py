'''
It gives different accuracy for each of the batches. Although it gives good result for each of the batches, specially
for initial batches, it has not a good performance after voting.
An ensemble of the basic feature extractor. Each model has been trained on a batch of data.
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
dataset = np.array(dataset).reshape((10, 500000, 3))

df_test = []
df_train = []
for batch in dataset:
    d = batch
    train_size = int(d.shape[0] * 0.75)
    df_train.append(pd.DataFrame(d[:train_size]))
    df_test.append(np.array(d[train_size:]))

# df_train = np.array(df_train)
# df_train = df_train.reshape((df_train.shape[0] * df_train.shape[1], df_train.shape[2]))
# df_train = pd.DataFrame(df_train)
#
df_test = np.array(df_test)
df_test = df_test.reshape((df_test.shape[0] * df_test.shape[1], df_test.shape[2]))
df_test = pd.DataFrame(df_test)


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


nb_model_per_batch = 1
model_pool = {i: None for i in range(nb_model_per_batch)}

for d in df_train:
    for j in range(nb_model_per_batch):
        data = d.sample(frac=0.7)
        x = data[1]
        y = data[2]
        model = create_model()
        model.fit(x, y, epochs=1, validation_split=.3, verbose=1)
        model_pool[j] = model

print('Done')
# predictions = model.predict(x)

nb_classes = 11
df_test = df_test.sample(frac=0.02)
x_test = df_test[1]
y_test = df_test[2]
ensemble_prediction = np.zeros((y_test.shape[0], nb_classes))
for j in range(nb_model_per_batch):
    prediction = model_pool[j].predict(x_test)
    # p = np.argmax(prediction)
    ensemble_prediction += prediction

voting_prediction = ensemble_prediction.argmax(axis=1)
acc = accuraccy(voting_prediction, y_test)
print('Test Accuracy: %.3f' % acc)
