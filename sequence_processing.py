import numpy as np

def prepareSeqs(all_seq, TIMESTEPS):
    prepared_seqs = []
    for seq in all_seq:
        if len(seq) <= TIMESTEPS:
            prepared_seqs.append(prepareSeq(seq, TIMESTEPS))
        else:
            prepared_seqs.append(seq)
    return prepared_seqs


def prepareSeq(seq, timesteps):
    fullSeq = []
    for i in range(0, timesteps - len(seq)+1):
        fullSeq.append(seq[0])

    fullSeq = fullSeq + seq
    return fullSeq


def splitToSubContext(seqs, timesteps, labels):
    data = []
    for seq in seqs:
        rnn_df = []
        for i in range(len(seq) - timesteps):
            if labels:
                try:
                    seq_list = [seq[i + timesteps]]
                    rnn_df.append(np.array(seq_list, dtype=np.float32))
                except AttributeError:
                    rnn_df.append(seq[i + timesteps])
            else:
                data_ = seq[i: i + timesteps]
                data_ = np.array(data_, dtype=np.float32)
                data_ = data_.reshape((timesteps, 1))
                rnn_df.append(data_)
        data.append(np.array(rnn_df, dtype=np.float32))
    return data

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split(data, val_size, test_size)
    return (splitToSubContext(df_train, time_steps, labels=labels),
            splitToSubContext(df_val, time_steps, labels=labels),
            splitToSubContext(df_test, time_steps, labels=labels))


def split_d(data, timesteps):
    train_x, val_x, test_x = prepare_data(data, timesteps)
    train_y, val_y, test_y = prepare_data(data, timesteps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def split(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data[:nval], data[nval:ntest], data[ntest:]

    return df_train, df_val, df_test

