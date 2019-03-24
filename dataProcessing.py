from __future__ import division
from __future__ import print_function

import math

from utils import *
import pickle

# Generate the file used in METIS


def genMETIS(support):
    vertices = len(support[0][0])
    npadj = np.zeros((vertices, vertices))
    edges = support[0][0]
    edge_num = 0
    for edge in edges:
        a = edge[0]
        b = edge[1]
        if a != b:
            npadj[a][b] = 1
            npadj[b][a] = 1

    for i in range(len(npadj[0])):
        for j in range(0, i + 1):
            if npadj[i][j] == 1:
                edge_num = edge_num + 1;

    print(edge_num)
    f = open("./graph", "w")
    f.write(str(vertices) + " " + str(edge_num) + "\n")
    for i in range(len(npadj[0])):
        for j in range(len(npadj[0])):
            if npadj[i][j] == 1:
                f.write(str(j + 1) + " ")
        f.write("\n")
    f.close()


def loadGraph(sub_num, support):
    filestr = "./graph.part." + str(sub_num)
    f = open(filestr, 'r')
    ver_info = f.read().splitlines()
    f.close()
    vertices = len(ver_info)
    sub_graphs = []
    sub_mask = []
    sub_train = []
    test_indices = []
    for i in range(sub_num):
        sub_graphs.append([])
        sub_train.append([])
    for i in range(vertices):
        subid = int(ver_info[i])
        sub_graphs[subid].append(i)
    for i in range(sub_num):
        train_num = math.floor(len(sub_graphs[i]) / 3)
        for j in range(len(sub_graphs[i])):
            if j < len(sub_graphs[i]) - train_num:
                sub_train[i].append(sub_graphs[i][j])
            else:
                test_indices.append(sub_graphs[i][j])
    for i in range(sub_num):
        sub_mask.append(sample_mask(sub_train[i], support[0][2][0]))
    test_mask = sample_mask(test_indices, support[0][2][0])
    return sub_mask, sub_graphs, test_mask


def saveSubSupport(support, sub_num, sub_graphs):
    edge_weight = support[0][1]
    edge_node = support[0][0]
    ver_num = support[0][2][0]
    sub_support = []

    for i in range(sub_num):
        tmp1 = []
        for k in range(sub_num):
            sub_edge_node = []
            sub_edge_weight = []
            for j in range(len(edge_node)):
                if (edge_node[j][0] in sub_graphs[i]) and (edge_node[j][1] in sub_graphs[k]):
                    sub_edge_node.append(edge_node[j])
                    sub_edge_weight.append(edge_weight[j])
            tmp = (np.array(sub_edge_node), np.array(sub_edge_weight), (ver_num, ver_num))
            tmp1.append(tmp)
        sub_support.append(tmp1)

    output = open(str(sub_num) + 'sub_support.pkl', 'wb')
    pickle.dump(sub_support, output)
    output.close()


