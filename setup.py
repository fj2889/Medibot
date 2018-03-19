# encoding=utf-8
import sample.jieba
import sample.insuranceqa_data as insuranceqa

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/1 ".join(seg_list))  # 全模式

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

# seg_list = jieba.cut_for_search1("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))

# 导入数据格式为词向量
pairs_vocab_data = insuranceqa.load_pairs_vocab()
pairs_train_data = insuranceqa.load_pairs_train()
pairs_test_data = insuranceqa.load_pairs_test()
pairs_valid_data = insuranceqa.load_pairs_valid()

# 导入数据格式为字符串
pool_train_data = insuranceqa.load_pool_train()
pool_test_data = insuranceqa.load_pool_test()
pool_valid_data = insuranceqa.load_pool_valid()
pool_answer_data = insuranceqa.load_pool_answers()
pass

# valid_data, test_data and train_data share the same properties
# for x in test_data:
#     print('index %s value: %s ++$++ %s ++$++ %s' %
#           (x['qid'], x['question'], x['utterance'], x['label']))

# vocab_data = insuranceqa.load_pairs_vocab()
# vocab_data['word2id']['UNKNOWN']
# vocab_data['id2word'][0]
# vocab_data['tf']
# vocab_data['total']
