from __future__ import print_function

from placenet import score
import numpy as np
from utils.np_function import argkmax, softmax


def get_label_name(label_id_list):
    label_name_list = list()
    for idx in label_id_list:
        label_name_list.append(score.label_list[idx])
    return label_name_list


def print_place(shot_idx, shot_start, shot_end, place_result_idx, place_result_label, place_result_prob):
    output = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(
        shot_idx, shot_start, shot_end, place_result_idx, place_result_label, place_result_prob)
    print(output)
    return output


class Place:
    net = score.net

    def __init__(self):
        self.score_list = None

    def classifier(self, frame, log=False):
        result = score.detect(frame, self.net)
        if self.score_list is None:
            self.score_list = result
        else:
            self.score_list = \
                np.concatenate((self.score_list, result), axis=0)
        idx, prob = argkmax(result[0])
        if log is True:
            print("#{0}\t{1}\t{2}".format(idx, get_label_name(idx), prob))
        return idx, prob

    def estimate(self, log=False):
        if self.score_list is None:
            idx = list()
            prob = None
        else:
            sum_idx, sum_prob, sum_result = self.__estimate_by_sum()
            mean_idx, mean_prob, mean_result = self.__estimate_by_mean()
            total_result = (sum_result + mean_result) / 2
            idx, prob = argkmax(total_result)
            self.score_list = None
        if log is True:
            print("Shot{0}\t{1}\t{2}".format(idx, get_label_name(idx), prob))

        return idx, prob

    def __estimate_by_sum(self):
        result = np.sum(self.score_list, axis=0) / self.score_list.shape[0]
        idx, prob = argkmax(result)
        return idx, prob, result

    def __estimate_by_mean(self):
        result = np.median(self.score_list, axis=0)
        result = result / np.sum(result)
        idx, prob = argkmax(result)
        return idx, prob, result

