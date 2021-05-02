# A basic DNN made to evaluate performance of more sophisticated NN's.
The first DNN I made for distinguishing tt and ll from MadGraph5/pythia/delphi generated particle collisions

Usage with hefaistos-server:
1. Install python package requirements
```
pip3 install -r requirements.txt
```
2. Load the relevant data from the .root files
```
python3 dataloader.py
```
3. Run program
```
python3 basicdnn.py
```


Dataloader loads the selected fj_ columns from the root files, creates dataframes and gives ll target 1 value, and tt target 0 value, and saves them as datacombined/llttmix.h5

Basicdnn creates a basic deep neural network, with optional normalizers, checkpoint saving and multigpu support.
If you dont want a certain data to be included, you can just comment it out with # in the NUMERIC_FEATURE_KEYS

## Running with docker
```
docker run -v "$(pwd):/basicdnn" \
           -v "/work/data/VBS/DNNTuplesAK8/:/work/data/VBS/DNNTuplesAK8/" \
           --gpus all -it --rm nvcr.io/nvidia/tensorflow:21.04-tf2-py3 bash
```
