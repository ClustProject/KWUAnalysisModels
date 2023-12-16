import numpy as np

def neighbors_verification(data, label, length, data_name):
    pos = np.where(label == 2)[0]
    neu = np.where(label == 1)[0]
    neg = np.where(label == 0)[0]
    pos_data, neu_data, neg_data = None, None, None
    if data.shape[1] != data.shape[0]:
        pos_data = data[pos[0]]
        neu_data = data[neg[0]]
        neg_data = data[neg[0]]
    else:
        pos_data = np.where(data[pos[0]] != 0)
        neu_data = np.where(data[neu[0]] != 0)
        neg_data = np.where(data[neg[0]] != 0)

    cnt = 0

    for i in range(length):
        if pos_data[i] in pos:
            cnt += 1
    print("The number of neighbors around a sample of {} in a class 'positive': {}".format(data_name, cnt))
    for i in range(length):
        if neu_data[i] in neu:
            cnt += 1
    print("The number of neighbors around a sample of {} in a class 'neutral': {}".format(data_name, cnt))
    for i in range(length):
        if neg_data[i] in neg:
            cnt += 1
    print("The number of neighbors around a sample of {} in a class 'negative': {}".format(data_name, cnt))

    return