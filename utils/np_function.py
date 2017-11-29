import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e = np.exp(x)
    dist = e / (np.sum(e) + 1e-8)
    return dist


def argkmax(data, top_number=5):
    top_number = np.minimum(top_number, len(data))
    if top_number == 0:
        return [], data
    idx = np.flip(np.argsort(data, axis=0), axis=0).tolist()[:top_number]
    prob = list()
    for i in idx:
        prob.append(data[i])
    # prob = np.zeros_like(data, dtype=np.float)
    # for i in idx:
    #     prob[i] = data[i]
    return idx, prob