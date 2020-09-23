#basic dnn testing for HEP with hardcoded data
#Niklas Harju
#18.6.2020
#

try:
    import setGPU #enforcing a single GPU, remove if using multiGPU model
    import os
    import itertools
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
except ModuleNotFoundError as err:
    print(err)

METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
NUMERIC_FEATURE_KEYS = [
    'fj_sdsj1_pt',  #
    'fj_sdsj2_pt',  #
    'fj_mass', #-
    'fj_tau1',  #
    'fj_tau21',  #-
    'fj_tau3',#
    'fj_eta',
    'fj_pt',
    'fj_relptdiff',  #
    'fj_sdn2',  #
]
BATCH_SIZE = 64
EPOCHS = 30
OPTIMIZER = Nadam(learning_rate=0.001)
#OPTIMIZER = SGD(learning_rate=0.001, nesterov=True, name='SGD')

def main():
    # load data with dataloader.py , columns are hard-coded
    df_copy = pd.read_hdf('datacombined/llttmix.h5', 'df')  #
    
    for combos in some_combos():

        # clear Keras memory for new layers, new column combinations from some_combos()
        tf.keras.backend.clear_session()


        print(combos)
        combos.append('target')

        df = df_copy[combos]
        
        TT, LL = np.bincount(df['target'])
        total = TT + LL
        print(f'Examples:\n    Total: {total}\n    LL: {LL} ({100 * LL / total}% of total)\n')

        #checking wanted columns are staying
        print(df.head)

        #df = normalize_minmax(df, NUMERIC_FEATURE_KEYS) #uncomment if you want to use min max norm, and comment normstd
        #df = normalize_tf(df, NUMERIC_FEATURE_KEYS)  #uncomment if you want to use tensorflow normalization

        df = df.sample(frac=1) #shuffle data
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
        print(class_weights)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=4,
            mode='max',
            restore_best_weights=True)
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
                  callbacks=[early_stopping]
                  )

        #model.save('datacombined/dnnmodel')
        model.summary()

        #test_loss, test_acc = model.evaluate(test_x, test_y)
        #print(f" Test accuracy: {test_acc}, test loss: {test_loss}")
        pred_y = model.predict(test_x)
        test_roc_auc = roc_auc_score(test_y, pred_y)
        print('Test set ROC AUC: ' + str(test_roc_auc))

        max_combos = ""
        max_roc = 0
        if test_roc_auc>max_roc:
            max_roc=test_roc_auc
            max_combos=combos
        with open('results.txt', 'a') as file:
             file.write(f'With {combos} we get: \n Test ROC AUC: {test_roc_auc}\n ')#\n and test loss: {test_loss} with test acc: {test_acc}')
        file.close()
        print(f'\n\n\n FINISHED TRAINING WITH:\n {combos}\n\n Saving to results.txt...\n')



    print(f"\n\nMax: {max_combos} with score {max_roc} \n\n")

def all_combos(nkeys = NUMERIC_FEATURE_KEYS):
     #some combinations tried with different columns.
     #Using all combinations is not recommended (range from 0 to len+1) as the number would be n!/(n-r)! (a lot of computation

    temp_array = []
    for L in range(len(nkeys)-1, len(nkeys)+1): 
        for subset in itertools.combinations(nkeys, L):
            if len(subset)==0:
                continue
            else:
                temp_array.append(list(subset))
    print(f"\nThe possible combinations are {temp_array}.\n")
    return temp_array

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
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dense(1, activation="sigmoid")
    ])
    return model

def multigpu_model(optimizer=OPTIMIZER):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f" \n\n {gpus} \n\n ")
    # tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()  # (devices=["/gpu:2", "/gpu:3"] ) would use two gpus at slots 2 and 3
    with strategy.scope():
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]

        model = model_layout()
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
    return model

def singlegpu_model(model, metrics=METRICS, optimizer=OPTIMIZER):
    
    model.compile(optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=metrics)
    return model


if __name__ == "__main__":
    main()


