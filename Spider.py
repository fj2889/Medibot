from bs4 import BeautifulSoup
import requests
import os
import progressbar
import csv
import re
import multiprocessing as mp
import functools


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
        self.bar = progressbar.ProgressBar(max_value=len(self.filespath))

    def file_process(self, file):
        try:
            with open(file, errors='ignore') as f:
                self.bar.update(self._bar_index)
                self._bar_index += 1

                file_content = f.read()

                soup = BeautifulSoup(file_content, 'lxml')

                # question = soup.select('#d_askH1')[0].string
                questiondetails = soup.select(
                    "#d_msCon")

                if len(questiondetails) == 0:
                    return
                accept_content = soup.select(
                    ".b_anscont_cont > .crazy_new")[0]
                if len(accept_content) == 0:
                    return

                questiondetails = questiondetails[0].get_text().replace(
                    ' ', ' ')
                questiondetails = re.sub(
                    self.re_double_empty_line, '\n', questiondetails)
                questiondetails = questiondetails.replace('\n', '<br>')

                q_des = re_match(
                    [r'健康咨询描述：(.*)曾经的治疗情况和效果：',
                     r'健康咨询描述：(.*)想得到怎样的帮助：',
                     r'健康咨询描述：(.*)'],
                    questiondetails)
                q_past = re_match(
                    [r'曾经的治疗情况和效果：(.*)想得到怎样的帮助：',
                     r'曾经的治疗情况和效果：(.*)'],
                    questiondetails)
                q_help = re_match([r'想得到怎样的帮助：(.*)'],
                                  questiondetails)

                if len(q_des) == 0 and len(q_past) == 0 and len(q_help) == 0:
                    q_des = questiondetails
                # if accept_content:
                accept_content = accept_content.get_text().replace(' ', ' ')
                accept_content = re.sub(
                    self.re_double_empty_line, '\n', accept_content)
                accept_content = accept_content.replace('\n', '<br>')
                a_analysis = re_match(
                    [r'病情分析：(.*)指导意见：',
                        r'病情分析：(.*)'],
                    accept_content)
                a_suggestion = re_match(
                    [r'指导意见：(.*)'],
                    accept_content)

                if len(a_analysis) == 0 and len(a_suggestion) == 0:
                    a_suggestion = re_match(
                        [r'(.*)'],
                        accept_content)
                result = [''.join(q_des), ''.join(q_past), ''.join(q_help), ''.join(
                    a_analysis), ''.join(a_suggestion)]

                self.csv_writer.writerow(result)
                # return result
                return
        except Exception as e:
            print(e)
            return

    def start(self):
        print('start process files...')
        with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
            processer = list(map(self.file_process, self.filespath))

        self.bar.finish()
        self.csvfile.close()


Folder_Path = r'/Users/yuxiangli/备份/网站资料库/www.120ask.com/question/'
spider = Spider(Folder_Path)
spider.start()
