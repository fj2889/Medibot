import pandas as pd
import numpy as np
import pickle
import re



def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# read tab-delimited file
with open('zh.tsv', 'r') as fin:
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

save_obj(id2word, 'id2word.pkl')
np.save('vec', vec)


word2vec = [[str(data[1])] + [float(x) for x in data[2:]]
            for data in filecontents if(len(data) != 0)]
word2vec = pd.DataFrame(word2vec)
np.save('word2vec', word2vec)
