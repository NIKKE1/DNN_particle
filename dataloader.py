#dataloader for basicdnn.py, hardcoded data paths
import os

try:
    import uproot
    import logging
    import pandas as pd
    import numpy as np
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

except ImportError as error:
    print("Installation of uproot missing...")

filePathll = '/work/data/VBS/DNNTuplesAK8/WWjj_SS_ll_hadronic.root'
filePathtt = '/work/data/VBS/DNNTuplesAK8/WWjj_SS_tt_hadronic.root'

# loading data
try:
    rootFilell = uproot.open(filePathll)['deepntuplizer']['tree']
    rootFilett = uproot.open(filePathtt)['deepntuplizer']['tree']
except ImportError:
    print("Unable to load data...")

print("Root files loaded, creating dataframes...")
df_ll = pd.DataFrame()
df_tt = pd.DataFrame()

# loading root columns to dataframes
print("Loading data into dataframes...")

keys = [
    'fj_sdsj1_pt',  #
    'fj_sdsj2_pt',  #
    'fj_mass',
    'fj_tau1',  #
    'fj_tau21',  #
    'fj_tau3', #
    'Delta_gen_pt',
    'fjJMAR_gen_pt',  #
    'fj_eta',
    'fj_gen_pt',  #
    'fj_pt',
    'fj_relptdiff',  #
    'fj_sdn2',  #
]

for key in keys:
    df_ll[key] = np.array(rootFilell.array(key))
    df_tt[key] = np.array(rootFilett.array(key))

df_ll['target'] = 1 # ll is what we want to find, so it will be the target
df_tt['target'] = 0

print(f"\nShape of df_ll: {df_ll.shape}\n")
print(f"\nShape of df_tt: {df_tt.shape}\n")

df = pd.concat([df_ll, df_tt]) #shuffling done by tensorflow

print(df.head)
print(f"\nShape of df combined: {df.shape}\n")


name = "llttmix.h5" 
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

data2h5(df, name, destination) #name is chosen keys+.h5 created in folder datacombined


