# =================================
# using for initialize data sets
# =================================
import sample.jieba as jieba
import sample.insuranceqa_data as insuranceqa
import logging
import progressbar


class dataset():
    _stopwordset = ''
    _splitsymbol = ''

    _counter = 0
    _bar = progressbar.ProgressBar(max_value=100)

    def __init__(self, *, splitsymbol, stopword_files):
        if(stopword_files != []):
            self.set_stopword(stopword_files)
        self._splitsymbol = splitsymbol

    def set_stopword(self, files):
        """
        load stop words
        """
        try:
            stopwords = ''
            for item in files:
                stopwords = ''.join(stopwords + '\n' + self.readfile(item))
            stopwordslist = stopwords.split('\n')
            not_stopword = set(['', '\n'])
            self._stopwordset = set(stopwordslist)
            self._stopwordset = self._stopwordset - not_stopword
        except Exception as e:
            print(e)

    def movestopwords(self, sentence):
        '''
        remove stop words
        '''

        try:
            def is_stopwords(word):
                if (word != '\t'and'\n') and (word not in self._stopwordset):
                    return word and word.strip()
            res = list(filter(is_stopwords, sentence))
        except Exception as e:
            print(e)

        return res

    def savefile(self, savepath, content):
        '''
        save files
        '''
        try:
            fp = open(savepath, 'w', encoding='utf8', errors='ignore')
            fp.write(content)
            fp.close()
        except Exception as e:
            print(e)

    def readfile(self, path):
        '''
        read files
        '''
        try:
            fp = open(path, "r", encoding='utf8', errors='ignore')
            content = fp.read()
            fp.close()
        except Exception as e:
            print(e)

        return content

    def _vectdata(self, question, answer, negative, domain, splitsymbol):
        """
        pair question and answer into one setence vector
            :param question:
            :param answer:
            :param negative:
            :param domain:
        """
        # split words
        try:
            seg_question = jieba.lcut(question)
            seg_answer = jieba.lcut(answer)

            if (self._stopwordset != []):
                seg_question = self.movestopwords(seg_question)
                seg_answer = self.movestopwords(seg_answer)
            result = [seg_question, seg_answer, negative, domain, splitsymbol]
            return result
        except Exception as e:
            print(e)

    def vectdata(self, data):
        self._counter = self._counter + 1
        self._bar.update(self._counter)
        question = data['zh']
        answer = data['str_answer']
        negative = data['negatives']
        domain = data['domain']
        splitsymbol = self._splitsymbol
        dataset = self._vectdata(
            question, answer, negative, domain, splitsymbol)
        return dataset

    def setbar(self, max):
        self._bar = progressbar.ProgressBar(max_value=max)
        self._counter = 0
