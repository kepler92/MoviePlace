from __future__ import print_function

from placenet import score
import numpy as np
from utils.np_function import argkmax, softmax

class Place:
    net = score.net

    def __init__(self):
        self.score_list = None

    def classifier(self, frame, frame_number=None):
        result = score.detect(frame, self.net)
        if self.score_list is None:
            self.score_list = result
        else:
            self.score_list = \
                np.concatenate((self.score_list, result), axis=0)
        idx, prob = argkmax(result[0])
        if frame_number is not None:
            print("#{0}\t{1}\t{2}\t{3}".format(frame_number, idx, score.get_label_name(idx), prob))
        return idx, prob

    def estimate(self, shot_number=None):
        if self.score_list is None:
            idx = list()
            prob = None
        else:
            result = np.sum(self.score_list, axis=0) / self.score_list.shape[0]
            self.score_list = None
            idx, prob = argkmax(result)
        if shot_number is not None:
            print("Shot{0}\t{1}\t{2}\t{3}".format(shot_number, idx, score.get_label_name(idx), prob))
        return idx, prob
