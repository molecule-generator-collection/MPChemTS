import csv
import itertools
import operator
import numpy as np
import nltk
import h5py
import os
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers import Dropout
import numpy as np
import random
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import model_from_json
# from tensorflow.keras.models import model_from_json
#from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket


def prepare_data(smiles, all_smile):
    all_smile_index = []
    for i in range(len(all_smile)):
        smile_index = []
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train = all_smile_index
    y_train = []
    for i in range(len(X_train)):

        x1 = X_train[i]
        x2 = x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train, y_train

def stateful_logp_model():
    vocab_size=64 #len(vocabulary)
    embed_size=64 #len(vocabulary)

    N=1 #X.shape[1]
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=vocab_size, batch_size=1, mask_zero=False))
    model.add(GRU(output_dim=256, batch_input_shape=(1,None,64),activation='sigmoid',return_sequences=True, stateful=True))
    model.add(GRU(256,activation='sigmoid',return_sequences=False, stateful=True))
    model.add(Dense(embed_size, activation='softmax'))
    model.load_weights('models/logpmodel/model.h5')
    return model


def loaded_logp_model():
#    json_file = open('models/logpmodel/model.json', 'r')
    json_file = open('models/logpmodel/model_nopad.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('models/logpmodel/model.h5')
    print("Loaded model from disk")
    return loaded_model


def loaded_wave_model():
    json_file = open('models/wavemodel/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('models/wavemodel/model.h5')
    print("Loaded model from disk")

    return loaded_model
