from __future__ import print_function

from placenet import score
import numpy as np
from utils.np_function import argkmax, softmax

class place:
    net = score.net

    def __init__(self):
        self.score_list = None

    def classifier(self, frame, log=None):
        result = score.detect(frame, self.net)
        if self.score_list is None:
            self.score_list = result
        else:
            self.score_list = \
                np.concatenate((self.score_list, result), axis=0)
        idx, prob = argkmax(result[0])
        if log is not None:
            print("#{0}\t{1}\t{2}".format(log, idx, prob), end=" ")
        return idx, prob

    def estimate(self, log=None):
        if self.score_list is None:
            idx = list()
            prob = None
        else:
            result = np.sum(self.score_list, axis=0) / self.score_list.shape[0]
            self.score_list = None
            idx, prob = argkmax(result)
        if log is not None:
            print("Shot{0}\t{1}\t{2}".format(log, idx, prob))
        return idx, prob
