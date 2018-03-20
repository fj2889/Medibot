# encoding=utf-8
import sample.jieba as jieba
import sample.insuranceqa_data as insuranceqa
import init
# from sklearn.feature_extraction.text import CountVectorizer

# 加载停用词表
stopwordset = init.load_stop_word(
    ['dict/哈工大停用词表.txt', 'dict/中文停用词.txt'])

# 导入数据集(数据格式为字符串)
pool_train_data = insuranceqa.load_pool_train()
pool_test_data = insuranceqa.load_pool_test()
pool_valid_data = insuranceqa.load_pool_valid()
pool_answer_data = insuranceqa.load_pool_answers()

# 分词
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")

pass
# 对句子去除停用词


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
