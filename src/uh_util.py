import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def read_matches_files(match_file_path):
    matches = []
    with open(match_file_path, 'r') as f:
        for line in f:
            line_split = line.split()
            matches.append([int(line_split[0]), int(line_split[3]),
                            int(line_split[1] == line_split[4])])
    return matches


def get_test_fea(test_fea, match_file_path):
    idm = read_matches_files(match_file_path)

    test_num = test_fea.shape[0]
    idxs = np.arange(0, test_num, 2)
    bidxs = idxs + 1

    test_fea_a = test_fea[idxs, :]
    test_fea_b = test_fea[bidxs, :]
    labels = [x[-1] for x in idm]

    return (test_fea_a, test_fea_b), labels


def get_batch_data(train_fea, indecies):
    cur_tr_fea = {}
    for k, v in train_fea.items():
        cur_tr_fea[k] = train_fea[k][indecies, :]
    return cur_tr_fea


def CompRealCode(rc_view_batch, pow_alpha):
    num_bit, num_sample = rc_view_batch[0].size()
    num_view = len(rc_view_batch)
    rc_final = torch.zeros(num_bit, num_sample).cuda()
    for jj in range(num_view):
        rc_final += pow_alpha[jj] * rc_view_batch[jj]
    rc_final /= torch.sum(pow_alpha)
    return rc_final


def CalDistance(code_matrix_a, code_matrix_b, mode='e'):
    if mode == 'e':
        diff = code_matrix_a - code_matrix_b
        dist_matrix = diff.mul(diff)
        dist = torch.sum(dist_matrix, 0)
        dist = torch.sqrt(dist)
        return dist
    elif mode == 'h':
        dist_matrix = torch.where(code_matrix_a == code_matrix_b, torch.zeros_like(code_matrix_a), torch.ones_like(code_matrix_a))
        dist = torch.sum(dist_matrix, 0).float()
        return dist
    else:
        print("[CalDistance] wrong mode!")


def SelectNegByCluster(sel_neg_num,cluster_dict,remove_list):
    n_clusters = len(cluster_dict.keys())
    sel_neg_indecies = []
    if sel_neg_num % n_clusters ==0:
        n_neg_per_clus = sel_neg_num//n_clusters

        for (k,v) in cluster_dict.items():
            v = set(np.random.permutation(v))
            remove_set = set(remove_list)
            new_v = list(v-remove_set)
            cur_npc = min(n_neg_per_clus, len(new_v))
            sel_neg_indecies.extend(new_v[:cur_npc])
    else:
        extra = sel_neg_num % n_clusters
        base_n_neg = sel_neg_num // n_clusters
        rand_clus = np.random.randint(0,n_clusters,size=extra)
        for i in range(n_clusters):
            v = set(np.random.permutation(cluster_dict[str(i)]))
            remove_set = set(remove_list)
            new_v = list(v - remove_set)

            if i in rand_clus:
                bnn = base_n_neg + 1
            else:
                bnn = base_n_neg
            bnn = min(bnn,len(new_v))
            sel_neg_indecies.extend(new_v[:bnn])
    sel_neg_indecies = np.random.permutation(sel_neg_indecies)

    return sel_neg_indecies



