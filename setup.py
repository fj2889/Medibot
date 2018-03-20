# encoding=utf-8
import sample.jieba as jieba
import sample.insuranceqa_data as insuranceqa
import init
import numpy as np
# import pandas as pd
# 这个包快一点
import ray.dataframe as pd
from multiprocessing.dummy import Pool as ThreadPool

# from sklearn.feature_extraction.text import CountVectorizer


dateset = init.dataset(splitsymbol='<POS>', stopword_files=[
                       'dict/哈工大停用词表.txt', 'dict/中文停用词.txt'])

# 初始化停用词表


# 导入数据集(数据格式为字符串)
print('导入数据集')
pool_train_data = insuranceqa.load_pool_train()
pool_test_data = insuranceqa.load_pool_test()
pool_valid_data = insuranceqa.load_pool_valid()
pool_answer_data = insuranceqa.load_pool_answers()


def pair_answer(question):
    answerid = question["answers"]

    def id2answer(id):
        return pool_answer_data[id]['zh']
    answer = list(map(id2answer, answerid))
    result = ''.join(answer)
    question['str_answer'] = result
    return question


# 创建线程池
print('\n\n训练集分词')
dateset.setbar(len(pool_train_data))
pool = ThreadPool(4)
pd_train_data = pool.map(pair_answer, pool_train_data.values())
pd_train_data = pool.map(dateset.vectdata, pd_train_data)
pd_train_data = pd.DataFrame(list(pd_train_data))
pool.close()
pool.join()

print('\n\n测试集分词')
pool = ThreadPool(4)
dateset.setbar(len(pool_test_data))
pd_test_data = pool.map(pair_answer, pool_test_data.values())
pd_test_data = pool.map(dateset.vectdata, pd_test_data)
pd_test_data = pd.DataFrame(list(pd_test_data))
pool.close()
pool.join()

print('\n\nCV集分词')
pool = ThreadPool(4)
dateset.setbar(len(pool_valid_data))
pd_valid_data = pool.map(pair_answer, pool_valid_data.values())
pd_valid_data = pool.map(dateset.vectdata, pd_valid_data)
pd_valid_data = pd.DataFrame(list(pd_valid_data))
pool.close()
pool.join()

pass

# 构建字典
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(train_data)


# valid_data, test_data and train_data share the same properties
# for x in test_data:
#     print('index %s value: %s ++$++ %s ++$++ %s' %
#           (x['qid'], x['question'], x['utterance'], x['label']))

# vocab_data = insuranceqa.load_pairs_vocab()
# vocab_data['word2id']['UNKNOWN']
# vocab_data['id2word'][0]
# vocab_data['tf']
# vocab_data['total']
