import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from igraph import *
from sklearn.metrics import mean_squared_error
from tensorflow.contrib import learn

from lstm import lstm_model
from sequence_processing import *

tf.logging.set_verbosity(tf.logging.INFO)
import os, shutil

LOG_DIR = './ops_logs/lstm_seqs'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 10}]
DENSE_LAYERS = None
USE_SUPERTYPES = False
TRAINING_STEPS = 500000
BATCH_SIZE = 1000
PRINT_STEPS = TRAINING_STEPS / 100
ALPHABET_PATH = "alphabet.txt"
DATA_PATH = "./data/"
#RNN_LAYERS = [{'num_units': 10}, {'num_units': 5}]
#DENSE_LAYERS = None
data_in = DATA_PATH + "test1.gml"
data_ascii = DATA_PATH + "test-no-ascii2.gml"

alphabet = {0: ""}
supertypes = {0: ""}
all_seq = []


def getSupertypeValue(intent):
    if intent == "":
        return 0
    for supertype, intentList in supertypes.items():
        if intent in intentList:
            return supertype


def traverse(graph, vertex, sequence):
    if sequence == None:
        current_seq = []
    else:
        current_seq = list(sequence)
    if USE_SUPERTYPES == False:
        intent_num_value = alphabet.get(vertex["intent"])
    else:
        intent_num_value = getSupertypeValue(vertex["intent"])
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


def splitToSupertypes(supertypesdict):
    st1 = {"а", "б", "в", "г", "д"}
    st2 = {"е", "ж", "з", "и", "к"}
    st3 = {"л", "м", "н", "о", "п"}
    st4 = {"р", "с", "т", "у", "ф"}
    st5 = {"х", "ц", "ч", "ш", "щ"}
    supertypesdict[1] = st1
    supertypesdict[2] = st2
    supertypesdict[3] = st3
    supertypesdict[4] = st4
    supertypesdict[5] = st5


def load_weather_frame(filename):
    # load the weather data and make a date
    data_raw = pd.read_csv(filename, dtype={'Time': str, 'Date': str})
    data_raw['WetBulbCelsius'] = data_raw['WetBulbCelsius'].astype(float)
    times = []
    for index, row in data_raw.iterrows():
        _t = datetime.time(int(row['Time'][:2]), int(row['Time'][:-2]), 0)  # 2153
        _d = datetime.datetime.strptime(row['Date'], "%Y%m%d")  # 20150905
        times.append(datetime.datetime.combine(_d, _t))

    data_raw['_time'] = pd.Series(times, index=data_raw.index)
    df = pd.DataFrame(data_raw, columns=['_time', 'WetBulbCelsius'])
    return df.set_index('_time')


def processSequences(all_seq, TIMESTEPS):
    prepared_seqs = prepareSeqs(all_seq, TIMESTEPS)
    return split_d(prepared_seqs, TIMESTEPS)


def mergeSeq(trainx):
    return np.concatenate(trainx)


def mergeSeqs(X, y):
    return dict(train=mergeSeq(X['train']), val=mergeSeq(X['val']), test=mergeSeq(X['test'])), dict(
        train=mergeSeq(y['train']), val=mergeSeq(y['val']), test=mergeSeq(y['test']))


def clearLogFolder():
    for the_file in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def main():
    clearLogFolder()
    load_russian_alphabet(ALPHABET_PATH, alphabet)
    if USE_SUPERTYPES:
        splitToSupertypes(supertypes)
    g = Graph()
    g = g.Read(data_in)
    get_all_seqs(g)

    if len(all_seq) > 0:
        print("Parsingsequences")

        X, y = processSequences(all_seq, TIMESTEPS)

        print("Merging sequences")
        Xm, ym = mergeSeqs(X, y)

        print("Create regressor")
        regressor = learn.SKCompat(learn.Estimator(
            model_fn=lstm_model(
                TIMESTEPS,
                RNN_LAYERS,
                DENSE_LAYERS, optimizer="Adam"
            ),
            model_dir=LOG_DIR
        ))

        # create a lstm instance and validation monitor
        validation_monitor = learn.monitors.ValidationMonitor(Xm['val'], ym['val'],
                                                              every_n_steps=PRINT_STEPS,
                                                              early_stopping_rounds=1000)

        print("fit regressor")

        regressor.fit(Xm['train'], ym['train'],
                      monitors=[validation_monitor],
                      batch_size=BATCH_SIZE,
                      steps=TRAINING_STEPS)

        print("predicting")

        predicted = regressor.predict(Xm['test'])
        # rmse = np.sqrt(((predicted - ym['test']) ** 2).mean(axis=0))

        score = mean_squared_error(predicted, ym['test'])
        hited = hitpoint(predicted, ym['test'])
        print("MSE: %f" % score)
        print("hitpoint:", hited)
        # print (predicted)


def hitpoint(predicted, test):
    hited = 0
    i = 0
    for predicion in predicted:
        roundedPredicion = int(math.ceil(predicion))
        testValue = test[i][0]
        if testValue == int(roundedPredicion):
            hited = hited + 1
        i = i + 1
    hitpoint = hited / len(predicted)
    return hitpoint


if __name__ == "__main__":
    main()
