import numpy as np

def Intersection(target, others):
    score = np.zeros(others.shape[0])
    for i in range(others.shape[0]):
        score[i] = np.sum(np.minimum(target, others[i]))
    return score

def Euclid(target, others):
    score = np.zeros(others.shape[0])
    for i in range(others.shape[0]):
        score[i] = np.sqrt(np.sum(np.square(others[i]-target)))
    return score

def Sort(data, inv):
    if inv: # 降順
        idx =np.argsort(-data)
    else: #昇順
        idx = np.argsort(data)
    return idx