import os

try:
    import uproot
    import logging
    import pandas as pd
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
# import uproot4


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

keys = ['fj_sdsj1_pt', 'fj_mass', 'fj_tau1', 'Delta_gen_pt', 'fjJMAR_gen_pt', 'fj_eta','fj_gen_pt','fj_pt','fj_relptdiff','fj_sdn2']



for key in keys:
    df_ll[key] = rootFilell.array(key)
    df_tt[key] = rootFilett.array(key)
   

df_ll['target'] = 1
df_tt['target'] = 0

df = pd.concat([df_ll, df_tt])

print(f"\nSampling dataframe... \n\n {df.head}")

df = df.sample(frac=1)

print(f"\nDataframe sampling worked? \n\n {df.head}")

name = "llttmix.h5" #just changed for this, the name became too long. Will make this nicer later
destination = "datacombined"

# data into .h5
def data2h5(df, name, destination):

    if not os.path.exists(destination):
        os.makedirs(destination)
    output = os.path.join(destination, name)
    logging.info(output)
    if os.path.exists(output):
        logging.warning('... file already exists ...')

    print(f"\nSaving dataframes into datacombined/{name} ...\n")
    df.to_hdf(output, key='df', mode='w')

data2h5(df, name, destination) #name is chosen keys+.h5 created in folder datacombined
