# basicdnn
Dnn for distinguishing tt and ll from CMS data.

usage with hefaistos-server:

pip3 install -r requirements.txt

python3 dataloader.py
python3 basicdnn.py


Dataloader loads the selected fj_ columns from the root files, creates dataframes and gives ll target 1 value, and tt target 0 value, and saves them as datacombined/llttmix.h5

Basicdnn creates a basic deep neural network, with optional normalizers, checkpoint saving and multigpu support.
If you dont want a certain data to be included, you can just comment it out # in the NUMERIC_FEATURE_KEYS
