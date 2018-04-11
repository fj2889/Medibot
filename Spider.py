from bs4 import BeautifulSoup
import requests
import os
import progressbar
import csv
import re
import multiprocessing as mp
import functools
import file_process as fp
from functools import reduce

error_char = {r' ': r' ',
              r'，': r',',
              r'。': r'.',
              r'；': r';',
              r'：': r':',
              r'“': r'\'',
              r'”': r'\'',
              r'‘': r'\'',
              r'’': r'\'',
              r"！": r"!",
              r"？": r"?",
              r"＠": r"@",
              r"＿": r"_",
              r"（": r"(",
              r"\t": r" ",
              r"）": r")"}


def re_match(rules, data):
    for rule in rules:
        match = re.search(rule, data)
        if match is None:
            continue
        else:
            item = match.groups(1)[0]
            item = re.sub(r'(^<br>)|(<br>$)|(^\ {1,})|(\ {1,}$)', '', item)
            item = re.sub(r'(^<br>)|(<br>$)|(^\ {1,})|(\ {1,}$)', '', item)
            return item
    return[]


class Spider():
    re_double_empty_line = None
    csvfile = None
    csv_writer = None
    filespath = []
    _bar_index = 0
    bar = None

    def __init__(self, Folder_Path):
        self.csvfile = open('bigdata/csv_test.csv', 'w')
        self.csv_writer = csv.writer(self.csvfile)
        self.re_double_empty_line = re.compile("\n{1,}")
        # 递归遍历当前目录和所有子目录的文件和目录
        for dirpath, dirs, files in os.walk(Folder_Path):
            for name in files:  # files保存的是所有的文件名
                if os.path.splitext(name)[1] == '.htm':
                    # 加上路径，dirpath是遍历时文件对应的路径
                    filename = os.path.join(dirpath, name)
                    self.filespath.append(filename)
                # if len(self.filespath) > 100:
                #     self.bar = progressbar.ProgressBar(
                #         max_value=len(self.filespath))
                #     return
        self.bar = progressbar.ProgressBar(max_value=len(self.filespath))

    def file_process(self, file):
        try:
            with open(file, errors='ignore') as f:
                if self._bar_index % 100 == 0:
                    self.bar.update(self._bar_index)
                self._bar_index += 1

                file_content = f.read()

                soup = BeautifulSoup(file_content, 'lxml')
                questiontitle = soup.select("#d_askH1")
                questiondetails = soup.select("#d_msCon")
                accept_content = soup.select(".b_anscont_cont > .crazy_new")
                if len(accept_content) == 0:
                    return

                if isinstance(questiontitle, list):
                    questiontitle = questiontitle[0].get_text()

                if isinstance(questiondetails, list):
                    questiondetails = questiondetails[0]

                if isinstance(accept_content, list):
                    accept_content = accept_content[0]

                def sample_process(data):
                    for source, substitute in error_char.items():
                        data = data.replace(source, substitute)

                    data = re.sub("\n{1,}", ' ', data)
                    data = re.sub(" {1,}", ' ', data)
                    data = data.rstrip().lstrip()
                    return data

                questiondetails = sample_process(questiondetails.get_text())
                accept_content = sample_process(accept_content.get_text())

                result = [file,
                          ''.join(questiontitle),
                          ''.join(questiondetails),
                          ''.join(accept_content)]

                self.csv_writer.writerow(result)
                return
        except Exception as e:
            print(e)
            return

    def start(self):
        print('start process files...')
        results = list(map(self.file_process, self.filespath))
        # for item in results:
        #     self.csv_writer.writerow(item)
        self.bar.finish()
        self.csvfile.close()


Folder_Path = r'/Users/yuxiangli/备份/网站资料库/www.120ask.com/question/'
spider = Spider(Folder_Path)
spider.start()
