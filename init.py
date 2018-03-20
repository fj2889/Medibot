'''
返回非空
'''


def not_empty(word):
    return word and word.strip()


'''
加载停用词列表
'''


def load_stop_word(files):
    stopwords = ''
    for item in files:
        stopwords = ''.join(stopwords + '\n' + readfile(item))
    stopwordslist = stopwords.split('\n')
    not_stopword = set(['', '\n'])
    stopwordset = set(stopwordslist)
    stopwordset = stopwordset - not_stopword
    return stopwordset


'''
去除停用词
'''


def movestopwords(sentence, stopwordset):
    def is_stopwords(word):
        if   (word != '\t'and'\n') and (word not in stopwordset):
            return word and word.strip()
    res = filter(is_stopwords, sentence)
    return res
#


'''
保存文件的函数
'''


def savefile(savepath, content):
    fp = open(savepath, 'w', encoding='utf8', errors='ignore')
    fp.write(content)
    fp.close()


'''
读取文件的函数
'''


def readfile(path):
    fp = open(path, "r", encoding='utf8', errors='ignore')
    content = fp.read()
    fp.close()
    return content
