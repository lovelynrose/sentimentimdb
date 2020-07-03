import re
import time

from data.resources import contractions
from data.parameters.datasetnames import PROCESSED
from sentiment.interface.readdata import ReadData
from sentiment.interface.savedata import SaveData


class ExpandContractions():

    def __init__(self, lis=None, res=None, inptext=None, choice='imdb'):
        self.lis = lis
        self.res = res
        self.text = inptext
        self.choice = choice

    def process_list(self):
        lis1 = []
        start_time = time.perf_counter()
        print("Expanding Contractions...")
        for i in range(len(self.lis)):
            # print(self.lis[i])
            exp_con = ExpandContractions(inptext=self.lis[i])
            temp = exp_con.expand_contractions()
            lis1.append(temp)
            del exp_con
        end_time = time.perf_counter()
        print("Completed Expanding Contractions : ", start_time - end_time)
        self.res = lis1

    # Input contractions.py file
    def expand_contractions(self, contraction_mapping=contractions.CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        # print(expand_match)
        # print(self.text)

        expanded_text = contractions_pattern.sub(expand_match, self.text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def perform_expand(self, filename):
        get = ReadData(filename, choice=self.choice)
        self.lis = get.readDataReview('0')
        self.res=self.lis
        self.process_list()
        SaveData(PROCESSED['contractions'], choice=self.choice).saveDataReview(self.res)