from unidecode import unidecode
from igraph import *
from data_processing import *
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from lstm import lstm_model
from tensorflow.contrib import learn

LOG_DIR = './ops_logs/lstm_seqs'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100
ALPHABET_PATH = "alphabet.txt"
DATA_PATH = "./data/"

data_in = DATA_PATH + "test1.gml"
data_ascii = DATA_PATH + "test-no-ascii2.gml"

alphabet = {0: ""}
all_seq = []


def traverse(graph, vertex, sequence):
    if sequence == None:
        current_seq = []
    else:
        current_seq = list(sequence)
    intent_num_value = alphabet.get(vertex["intent"])
    if intent_num_value == None:
        intent_num_value = 0
    current_seq.append(intent_num_value)
    neightbors = graph.neighbors(vertex.index, mode='OUT')
    if len(neightbors) > 0:
        for n in neightbors:
            neightbor = graph.vs[n]
            traverse(graph, neightbor, current_seq)
    else:
        all_seq.append(current_seq)


def get_all_seqs(graph):
    traverse(graph, graph.vs[0], None)
    # print(all_seq)


def load_russian_alphabet(path, dict):
    with open(path, 'r', encoding='utf8') as alphabet_file:
        i = 1.0
        for line in alphabet_file:
            line = line.rstrip('\n')
            dict[line] = i
            i = i + 1
    # print(dict)


def split_d(data, timesteps):
    train_x, val_x, test_x = split(data)
    train_y, val_y, test_y = split(data)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def split(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data[:nval], data[nval:ntest], data[ntest:]

    return df_train, df_val, df_test


def main():
    load_russian_alphabet(ALPHABET_PATH, alphabet)
    g = Graph()
    g = g.Read(data_in)
    all_vertices = g.vs
    all_edges = g.es
    #X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)
    # print(X)
    get_all_seqs(g)

    if len(all_seq) > 0:
        X, y = split_d(all_seq, TIMESTEPS)

        regressor = learn.SKCompat(learn.Estimator(
            model_fn=lstm_model(
                TIMESTEPS,
                RNN_LAYERS,
                DENSE_LAYERS
            ),
            model_dir=LOG_DIR
        ))


    # create a lstm instance and validation monitor
    validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)
    regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

    predicted = regressor.predict(X['test'])

if __name__ == "__main__":
    main()
