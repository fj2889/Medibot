import csv
import pandas as pd
import numpy as np
import pickle
import os
from functools import reduce
import re
import multiprocessing as mp


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# read tab-delimited file
with open('/Users/yuxiangli/OneDrive - Landian Office 365/大学/毕设代码/Chatbot-code/sample/word2vec/source/zh.tsv', 'r') as fin:
    filecontents = fin.read()
    filecontents = filecontents.replace('\n', '').replace('[', '').split(']')
    filecontents = [list(filter(None, re.split('\t| ', str)))
                    for str in filecontents]
    filecontents = filecontents[0:-1]


content = pd.DataFrame(filecontents)
id2word = {str(data[1]): int(data[0])
           for data in filecontents if (len(data) != 0)}

vec = content.iloc[:, 2:]
vec = vec.rename(lambda x: str(int(x)-2), axis='columns')
vec = np.array(vec, dtype='float32')

save_obj(id2word, 'sample/word2vec/id2word.pkl')
np.save('sample/word2vec/vec.npz', vec)


word2vec = [[str(data[1])] + [float(x) for x in data[2:]]
            for data in filecontents if(len(data) != 0)]
word2vec = pd.DataFrame(word2vec)
np.save('data/word2vec.npz', word2vec)
