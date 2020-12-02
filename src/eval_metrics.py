import numpy as np

# checked
def ErrorRateAt95Recall(labels, dists):
    recall_point = 0.95
    sorted_dists = list(zip(labels, dists))
    sorted_dists = sorted(sorted_dists, key=lambda x: x[1])  # default : increase

    n_thresh = recall_point * np.sum(labels)
    # print("overall {} match".format(np.sum(labels)))

    TP = 0
    count = 0
    for label, _ in sorted_dists:
        count += 1
        if label == 1:
            TP += 1
        if TP >= n_thresh:
            break
    FP = float(count - TP)
    return FP / count

# not used
# FPRat95%Recall : [when (pre=1) number > 95% (GT=1), cal FPR]
def FPRAt95Recall(labels, distances):
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    print("sorted labels samples {}".format(labels[1000:1020]))
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    # FP+TN = (GT=0) number
    FP = np.sum(labels[:threshold_index] == 0)
    TN = np.sum(labels[threshold_index:] == 0)
    print("thre {} ; FP {}; TN {}".format(threshold_index, FP, TN))

    return float(FP) / float(FP + TN)
