import re
import time

from data.parameters.datasetnames import PROCESSED
from sentiment.interface.readdata import ReadData
from sentiment.interface.savedata import SaveData

# Remove Special Characters
# Input : text
# Output : Text After removing all special characters
class RemoveSpcChar:
    def __init__(self, lis=None, res=None, inptext=None, choice = 'imdb'):
        self.lis = lis
        self.res = res
        self.text = inptext
        self.choice = choice

    def remove_special_characters(self, remove_digits=True):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', self.text)
        return text

    def process_list(self):
        lis1 = []
        start_time = time.perf_counter()
        print("Remove Special Characters")
        for i in range(len(self.lis)):
            remspc = RemoveSpcChar(inptext=self.lis[i], choice=self.choice)
            temp = remspc.remove_special_characters()
            lis1.append(temp)
            del remspc
        end_time = time.perf_counter()
        print("Completed Removing Special Characters : ", start_time - end_time)
        self.res = lis1

    def perform_expand(self, filename):
        get = ReadData(filename, choice=self.choice)
        self.lis = get.readDataReview('0')
        # print(self.lis[18])
        self.process_list()
        SaveData(PROCESSED['spl_char'], choice=self.choice).saveDataReview(self.res)
