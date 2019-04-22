from __future__ import division
from __future__ import print_function

import math

from utils import *
import pickle

# Generate the file used in METIS


def genMETIS(dataset, support):
    vertices = support[0][2][0]
    print(support)
    edges = support[0][0]
    edge_num = 0
    neighbor = []
    for i in range(vertices):
        neighbor.append([])

    for edge in edges:
        a = edge[0]
        b = edge[1]
        if a != b:
            edge_num = edge_num + 1
            neighbor[a].append(b)

    f = open("./" + dataset, "w")
    f.write(str(vertices) + " " + str(edge_num) + "\n")
    for i in range(vertices):
        for j in range(len(neighbor[i])):
            f.write(str(neighbor[i][j] + 1) + " ")
        f.write("\n")
    f.close()


def loadGraph(dataset, sub_num, support):
    filestr = "./" + dataset + ".part." + str(sub_num)
    f = open(filestr, 'r')
    ver_info = f.read().splitlines()
    f.close()
    vertices = len(ver_info)
    sub_graphs = []
    sub_mask = []
    sub_train = []
    test_indices = []
    train_indices = []
    sub_reorder = []
    cnt = 0
    for i in range(sub_num):
        sub_graphs.append([])
        sub_train.append([])

    for i in range(vertices):
        subid = int(ver_info[i])
        sub_graphs[subid].append(i)
        sub_reorder.append(0)
    for i in range(sub_num):
        train_num = math.floor(len(sub_graphs[i]) / 3)
        for j in range(len(sub_graphs[i])):
            sub_reorder[cnt] = sub_graphs[i][j]
            cnt = cnt + 1
            if j < len(sub_graphs[i]) - train_num:
                sub_train[i].append(sub_graphs[i][j])
                train_indices.append(sub_graphs[i][j])
            else:
                test_indices.append(sub_graphs[i][j])

    for i in range(sub_num):
        for j in range(len(sub_graphs[i])):
            sub_graphs[i][j] = sub_reorder.index(sub_graphs[i][j])

    for i in range(sub_num):
        sub_mask.append(sample_mask(sub_train[i], support[0][2][0]))
    test_mask = sample_mask(test_indices, support[0][2][0])
    test_mask = [test_mask[i] for i in sub_reorder]

    train_mask = sample_mask(train_indices, support[0][2][0])
    train_mask = [train_mask[i] for i in sub_reorder]

    return sub_mask, sub_graphs, test_mask, train_mask, sub_reorder


def reorder_sup_feat(dataset, sub_num, support, features, sub_reorder):
    edge_node = support[0][0]
    resupport = []

    for i in range(len(edge_node)):
        v1 = edge_node[i][0]
        v2 = edge_node[i][1]
        edge_node[i] = [sub_reorder.index(v1), sub_reorder.index(v2)]

    resupport.append((edge_node, support[0][1], support[0][2]))
    feat_node = features[0]
    for i in range(len(feat_node)):
        feat_node[i][0] = sub_reorder.index(feat_node[i][0])
    refeatures = (feat_node, features[1], features[2])
    res = [resupport, refeatures]
    output1 = open(str(sub_num) + '.' + dataset + '.reorder_res.pkl', 'wb')
    pickle.dump(res, output1)
    output1.close()


def saveInnerSupport(dataset, support, sub_num, sub_graphs):
    edge_weight = support[0][1]
    edge_node = support[0][0]
    ver_num = support[0][2][0]
    sub_edge_node = []
    sub_edge_weight = []

    for i in range(sub_num):
        for j in range(len(edge_node)):
            if (edge_node[j][0] in sub_graphs[i]) and (edge_node[j][1] in sub_graphs[i]):
                sub_edge_node.append(edge_node[j])
                sub_edge_weight.append(edge_weight[j])

    inner_support = [(np.array(sub_edge_node), np.array(sub_edge_weight), (ver_num, ver_num))]
    output = open(str(sub_num) + '.' + dataset + '.inner_support.pkl', 'wb')
    pickle.dump(inner_support, output)
    output.close()


