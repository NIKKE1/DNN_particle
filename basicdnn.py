"""
try:
    #import setGPU

except ImportError:
    print("Import Error, do you have tensorflow_transform?")
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split

df = pd.read_hdf('datacombined/llttmix.h5', 'df') #
"""
NUMERIC_FEATURE_KEYS = [
    'fj_sdsj1_pt',
    'fj_mass',
    'fj_tau1',
]
for key in NUMERIC_FEATURE_KEYS:

    print(df[key].head)
    df[key] = tf.keras.utils.normalize(df[key], axis=2, order=2)
"""


print(df.dtypes)
print(df.head)
"""
target = df.pop('target')
print(target.head)

df = tf.keras.utils.normalize(df, axis=1, order=2) #euclidean normalization along columns, is this correct?
print(df)
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
"""

train, test = train_test_split(df, test_size=0.2, random_state=42)

train_y = train.target
test_y = test.target
train.pop('target')
test.pop('target')
train_x = train
test_x = test
print(train_x.head)
# Normalize the features
train_x_mean = train_x.mean()
train_x_std = train_x.std()

train_x = (train_x - train_x_mean) / train_x_std
test_x = (test_x - train_x_mean) / train_x_std  #this is not normalized to 1 and has negatives, issues with ReLU?



print(train_y.head)
print(train_x.head)
print(train_x_std.head)


def leakyrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

#build model
model = tf.keras.Sequential([
    Dense(256, activation = 'relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer='Nadam', loss = "binary_crossentropy", metrics=['accuracy'])
model.fit(train_x,
          train_y,
          validation_split=0.15,
          verbose=1,
          shuffle=True,
          epochs=2)

test_loss, test_acc = model.evaluate(test_x, test_y)
print(test_acc)

pred_y = model.predict(test_x)

plt.clf()
binning = np.arange(0.0, 1.0, 0.04)
plt.hist( pred_y[test_y==0], bins=binning, alpha=0.8, label="tt", density=1 )
plt.hist( pred_y[test_y==1], bins=binning, alpha=0.8, label="ll", density=1 )
plt.legend()
plt.xlabel('DNN output value')
plt.title('Simple DNN classifier')

