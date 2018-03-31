# encoding=utf-8

import init


# from sklearn.feature_extraction.text import CountVectorizer

dateset = init.Dataset(filename="corpus/corpus.csv",
                       splitsymbol='<POS>',
                       word2id_file='sample/word2vec/id2word.pkl',
                       id2vec_file='sample/word2vec/vec.npz.npy',
                       user_dict="dict/自定义词典.txt",
                       stopword_dict=[
                           'dict/哈工大停用词表.txt',
                           #    'dict/中文停用词.txt',
                           'dict/自定义停用词.txt'],
                       prop=[0.6, 0.2, 0.2])

dateset.save_obj('save.pkl', dateset)

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
