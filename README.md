# The VoxCeleb Speaker Recognition Challenge 2020 (VoxSRC-20)

Implementation of several loss functions and models to train speaker recognition model on VoxCeleb dataset.


## Data Preparation
Extract the data and create a csv formated file as follows:

```
ID0 /full/path/to/wav
ID1 /full/path/to/wav
...
```


## Train With Your Custom Setup

The code is modular such that one could combine desired trunk model and polling layer, then train the network with desired criterion:
```
python3.8 trainer.py --csv-path /path/to/csv --trunk-net resnet --lr 0.003 --batch-size 64 --polling-net tap --criterion cosface --m 0.1 --s 20 --criterion-lr 0.001
```

Take a look at `opts.py` to see the full options.

### Trunk Models
```
resnet34 (fewer stirdes)
resnet34se
```

### Polling Layers
```
tap
sap
```

### Criterions
```
cosface
psge2e*
protypical
```
*psge2e (pseudo ge2e loss): Despite the original version, it learns the speakers representations.

This repo is under heavy construction and the lists will be grown.
