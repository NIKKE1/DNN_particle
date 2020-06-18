basic dnn testing for HEP, hardcoded data
#Niklas Harju
#18.6.2020
#

try:
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import Nadam, SGD, Adam
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    print("Module import error: check that requirement modules are installed")

NUMERIC_FEATURE_KEYS = [
    'fj_sdsj1_pt',  #
    'fj_sdsj2_pt',  #
    'fj_mass',
    'fj_tau1',  #
    'fj_tau21',  #
    'fj_tau3',#
    'fj_eta',
    'fj_pt',
    'fj_relptdiff',  #
    'fj_sdn2',  #
    'target'
]
BATCH_SIZE = 32
EPOCHS = 100
OPTIMIZER = Nadam(learning_rate=0.0001)
#OPTIMIZER = SGD(learning_rate=0.001, nesterov=True, name='SGD')

def main():
    # load data with dataloader.py , columns are hard-coded
    df = pd.read_hdf('datacombined/llttmix.h5', 'df')  #

    #drop fj's that you dont want to inspect
    columns_to_drop = [key for key in df.columns if key not in NUMERIC_FEATURE_KEYS ]
    df.drop(df[columns_to_drop], axis=1, inplace=True)

    #checking wanted columns are staying
    print(df.head)

    #df = normalize_minmax(df, NUMERIC_FEATURE_KEYS) #uncomment if you want to use min max norm and comment normstd
    #df = normalize_tf(df, NUMERIC_FEATURE_KEYS)

    print(df.head)
    #slicing dataset for training and testing.

    train, test = train_test_split(df, test_size=0.2)
    train_y = train.target
    test_y = test.target
    train.pop('target')
    test.pop('target')
    train_x = train
    test_x = test
    train_x, test_x = normalize_std(train_x, test_x) #comment this out if you are using normalize_minmax

    # Weight the training samples so that there is equal weight on ll and tt
    # even if there are different amount of them in the training set

    class_weights = compute_class_weight('balanced', np.unique(train_y), train_y[:])
    class_weights = {i: class_weights[i] for i in range(2)}

    #singlegpu_model to enforce a single gpu, multigpu for tensorflows multi
    model = singlegpu_model(model_layout())

   #saving checkpoints between epochs

   #checkpoint_path = "models/cp.ckpt"  #Saving commented out since it seems to produce bugs occasionally
   #checkpoint_dir = os.path.dirname(checkpoint_path)
   #cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(train_x,
              train_y,
              validation_split=0.15,
              verbose=1,
              shuffle=True,
              batch_size=BATCH_SIZE,
              class_weight=class_weights,
              epochs=EPOCHS,
              #callbacks=[cp_callback]
              )

    #model.save('datacombined/dnnmodel')
    model.summary()

    test_loss, test_acc = model.evaluate(test_x, test_y)
    print(f" Test accuracy: {test_acc}, test loss: {test_loss}")
    pred_y = model.predict(test_x)
    test_roc_auc = roc_auc_score(test_y, pred_y)
    print('Test set ROC AUC: ' + str(test_roc_auc))


# Normalize the features options: std, tensorflow norm, minmax(0-1).

def normalize_std(train_x, test_x):
    train_x_mean = train_x.mean()
    train_x_std = train_x.std()

    train_x = (train_x - train_x_mean) / train_x_std
    test_x = (test_x - train_x_mean) / train_x_std  # this is not normalized to 1 and has negatives, issues with ReLU?
    return train_x, test_x

def normalize_tf(df, NUMERIC_FEATURE_KEYS):
    NUMERIC_FEATURE_KEYS.remove('target')
    for key in NUMERIC_FEATURE_KEYS:
        temp = np.array(df[key])
        temp = temp.reshape(-1, 1)
        temp = tf.keras.utils.normalize(temp, axis=0, order=2) #alternate tf norm includes -values
        temp.reshape(-1, )
        df[key] = temp
    return df

def normalize_minmax(df, NUMERIC_FEATURE_KEYS):
    scaler = MinMaxScaler()
    NUMERIC_FEATURE_KEYS.remove('target')
    for key in NUMERIC_FEATURE_KEYS:
        temp = np.array(df[key])
        temp = temp.reshape(-1, 1)
        scaler.fit(temp)
        scaler.data_max_
        temp = scaler.transform(temp)
        temp.reshape(-1, )
        
df[key] = temp
    return df

#leakyrelu could be tried instead of relu
def leakyrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def model_layout():
    model = tf.keras.Sequential([
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dense(1, activation="sigmoid")
    ])
    return model

def multigpu_model(model):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f" \n\n {gpus} \n\n ")
    # tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()  # (devices=["/gpu:2", "/gpu:3"] ) would use two gpus at slots 2 and 3
    with strategy.scope():
        model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=['accuracy'])
    return model

def singlegpu_model(model):
    import setGPU
    model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=['accuracy'])

if __name__ == "__main__":
    main()

""" Slicing with tf:
target = df.pop('target')
df = tf.keras.utils.normalize(df, axis=1, order=2) #euclidean normalization along columns, is this correct?
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
"""

"""
Bugs:  multigpu
ValueError: SyncOnReadVariable does not support `assign_add` 
in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`.

No idea what causes this but, but data-reload fixed it for me.9
"""


