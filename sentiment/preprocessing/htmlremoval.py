import re
import time

from sentiment.interface.savedata import SaveData
from sentiment.interface.readdata import ReadData


class HTMLTagRemoval:

    def __init__(self, raw_html=None, lis=None, res=None, sen=None, choice='imdb'):
        self.lis = lis
        self.res = res
        self.sen = sen
        self.raw_html = raw_html
        self.choice = choice


    def cleanhtml(self):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', self.raw_html)
        return cleantext

    def htmlTagRemoval(self):
        lis1 = []
        start_time = time.perf_counter()
        print("Removing HTML Tags...")
        for i in range(len(self.lis)):
            remove_html = HTMLTagRemoval(self.lis[i])
            temp = remove_html.cleanhtml()
            lis1.append(temp)
            del remove_html
        end_time = time.perf_counter()
        print("Completed Removing HTML Tags : ", -(start_time - end_time))
        self.res = lis1

    def perform_removal(self, filename):
        get = SaveData(filename, choice=self.choice)
        self.lis, self.sen = ReadData(filename, choice=self.choice).readData()
        self.htmlTagRemoval()
        get.saveData(self.res, self.sen)