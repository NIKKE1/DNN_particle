# Hardcoded data-loading for the basic DNN
# NH
# 28.9.2020
import os

try:
    import uproot3 as uproot
    import logging
    import pandas as pd
    import numpy as np
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
# import uproot4


except ImportError as error:
    print("Installation of uproot missing...")

filePathll = '/work/data/VBS/DNNTuplesAK8/WWjj_SS_ll_hadronic.root'
filePathll2='/work/data/VBS/DNNTuplesAK8/WWjj_SS_ll_ext_hadronic.root'
filePathtt = '/work/data/VBS/DNNTuplesAK8/WWjj_SS_tt_hadronic.root'
filePathll3 = '/work/data/VBS/DNNTuplesAK8/WWjj_ll_hadronic_HT1000Inf.root'
filePathtt2 = '/work/data/VBS/DNNTuplesAK8/WWjj_tt_hadronic_HT1000Inf.root'
# loading data
try:
    rootFilell = uproot.open(filePathll)['deepntuplizer']['tree']
    rootFilell2 = uproot.open(filePathll2)['deepntuplizer']['tree']
    rootFilell3 = uproot.open(filePathll3)['deepntuplizer']['tree']
    rootFilett = uproot.open(filePathtt)['deepntuplizer']['tree']
    rootFilett2 = uproot.open(filePathtt2)['deepntuplizer']['tree']
except ImportError:
    print("Unable to load data...")

print("Root files loaded, creating dataframes...")
df_ll = pd.DataFrame()
df_ll2= pd.DataFrame()
df_ll3 = pd.DataFrame()
df_tt = pd.DataFrame()
df_tt2 = pd.DataFrame()


# loading root columns to dataframes
print("Loading data into dataframes...")

keys = [
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
for key in keys:
    df_ll[key] = np.array(rootFilell.array(key))
    df_ll2[key] = np.array(rootFilell2.array(key))
    df_ll3[key] = np.array(rootFilell3.array(key))
    df_tt[key] = np.array(rootFilett.array(key))
    df_tt2[key] = np.array(rootFilett2.array(key))

print(df_ll3['fj_pt'])    
print(f'Slicing test dfll3 before slice {df_ll3.shape}')    
df_ll3 = df_ll3.loc[(df_ll3['fj_mass'] >= 60) & (df_ll3['fj_mass'] <= 100)]
df_tt2 = df_tt2.loc[(df_tt2['fj_mass'] >= 60) & (df_tt2['fj_mass'] <= 100)]
print(f'dfll3 after slice {df_ll3.shape}')

df_ll['target'] = 1
df_ll2['target'] = 1
df_ll3['target'] = 1
df_tt['target'] = 0
df_tt2['target'] = 0

print(f"\nShape of df_ll: {df_ll.shape}")
print(f"\nShape of df_ll2: {df_ll2.shape}")
print(f"\nShape of df_ll3: {df_ll3.shape}")
print(f"\nShape of df_tt: {df_tt.shape}")
print(f"\nShape of df_tt2: {df_tt2.shape}")

dftempll = [df_ll, df_ll2, df_ll3]
dftemptt = [df_tt, df_tt2]

dfll = pd.concat(dftempll)
dftt = pd.concat(dftemptt)

dfll = dfll.sample(frac=1)
dftt = dftt.sample(frac=1)


train_LL, val_LL, test_LL = np.split(dfll.sample(frac=1), [int(.7*len(dfll)), int(.85*len(dfll))])
train_TT, val_TT, test_TT = np.split(dftt.sample(frac=1), [int(.7*len(dftt)), int(.85*len(dftt))])

train = pd.concat([train_LL, train_TT])
val = pd.concat([val_LL, val_TT])
test = pd.concat([test_LL, test_TT])

destination = "datacombined"

# data into .h5
def data2h5(df, name, destination):

    if not os.path.exists(destination):
        os.makedirs(destination)
    output = os.path.join(destination, name)
    logging.info(output)
    if os.path.exists(output):
        logging.warning(' File already exists: OVERWRITING ...')

    print(f"\nSaving dataframes into datacombined/{name} ...\n")
    df.to_hdf(output, key='df', mode='w')


#train
data2h5(train, 'train.h5', destination) 
#test
data2h5(test, 'test.h5', destination)
#val
data2h5(val, 'val.h5', destination)


