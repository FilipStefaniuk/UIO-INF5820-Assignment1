#### Filip Stefaniuk
# Assignment 1: Neural Classicifaction with Bag of Words
## Overview
Program trains simple feed-forward neural network which takes bag-of-words text representations as input and produces source predictions as output. Data is loaded from compressed file and split into three parts: train, test and validation. Neural network model is build and trained according to setting in provided configuration, performance is measured on validation and possibly test set using such metrics as: accuracy, precision, recall and f1 score. Log from training and results of the evaluation are saved to json files.

## Running Program
Program requires one argument which is path to data. Additionally it is possible to pass path to the file with configuration, output directory and flag whether to use test data.
```
usage: main.py [-h] -d DATA [-o OUTPUT] [-c CONFIG] [-t]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to data file
  -o OUTPUT, --output OUTPUT
                        output directory
  -c CONFIG, --config CONFIG
                        path to json config file
  -t, --test            evaluate model on test dataset
```

## Configuration
Program takes configuration in json file format. Possible parameters are:
* seed - seed used by the random number generator. Used to shuffle the data (default 123).
* pos - whether to use words with POS tags in BOW (default True).
* test_size - float between 0 and 1, size of test data (default 0.1).
* val_size - float between 0 and 1, size of validation data (default 0.1).
* mode - mode in which tokenizer creates BOWs, can be: binary, count, tfidf, freq (default binary). 
* input_size - size of input vector, most n words will be used in BOW (default 100).
* loss - loss function used in model (default 'categorical_crossentropy')
* optimizer - optimizer used when training model (default 'adam').
here are possible values in those dictionaries:
  - type - required, one of the three types dense, batch_norm or dropout
  - units - relevant only with type=dense, number of units in the layer
  - activation - relevant only with type=dense, activation function.
  - regularization - relevant only with type=dense, regularization l1 or l2.
  - rate - relevant only with type=dropout, dropout rate
* epochs - number of epochs (default 1).
* early_stopping - whether to use early stopping (default False)
* patience - epochs with worse performance befor early stopping, relevant only with early stopping (default 0).
* weight_sample - whether to use weighted loss function
* layers - list of dictionaries that describes layers used in model