# Run if using tensorflow2.0+
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('../Data/train.csv')

ax = df_train['open_channels'].plot.hist(bins=11, alpha=0.5)
plt.title('Class histogram')
plt.show()

for i in range(0, 11):
    data = df_train[df_train.open_channels == i]
    plt.plot(data.signal, label=str(i))
plt.legend()
plt.show()

fig = plt.figure()
for i in range(0, 6):
    data = df_train[df_train.open_channels == i]
    ax = fig.add_subplot(3, 2, i + 1)
    plt.plot(data.signal, label=str(i))
    plt.title('Class: %d' % i)
plt.show()

fig = plt.figure()
for i in range(6, 11):
    data = df_train[df_train.open_channels == i]
    ax = fig.add_subplot(3, 2, i - 5)
    plt.plot(data.signal, label=str(i))
    plt.title('Class: %d' % i)
plt.show()

print('Done')
