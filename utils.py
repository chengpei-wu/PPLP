import dgl
import networkx as nx
import numpy as np
from dgl.data import *
from sklearn.preprocessing import OneHotEncoder

from PPLP import graph2vec


# load TUDataset
def load_dgl_data(dataset, pooling_sizes, rank_label, pooling_attr, pooling_way):
    enc = OneHotEncoder()
    data = TUDataset(dataset)
    x = []
    labels = []
    has_node_attr = 'node_attr' in data[0][0].nodes[0][0].keys()
    has_node_label = 'node_labels' in data[0][0].nodes[0][0].keys()
    if has_node_attr:
        num_node_attr = len(data[0][0].nodes[0][0]['node_attr'].numpy().flatten())
        print(f'Number of Node Attributes: {num_node_attr}')
    if has_node_label:
        print('Graph Has Node Labels')
    for id in range(len(data)):
        print('\r',
              f'loading {id} / {len(data)}  network...',
              end='',
              flush=True)
        graph, label = data[id]
        G = nx.Graph(dgl.to_networkx(graph))
        if has_node_attr:
            for i in range(G.number_of_nodes()):
                G.nodes[i]['node_attr'] = graph.nodes[i][0]['node_attr'].numpy().flatten()
        if has_node_label:
            for i in range(G.number_of_nodes()):
                G.nodes[i]['label'] = graph.nodes[i][0]['node_labels'].numpy().flatten()
        x.append(
            graph2vec(
                G=G,
                rank_label=rank_label,
                pooling_sizes=pooling_sizes,
                pooling_attr=pooling_attr,
                pooling_way=pooling_way
            )
        )
        labels.append(label.numpy())
    y = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    x = np.array(x)
    return x, y


def print_progress(now, total, length=20, prefix='progress:'):
    print('\r' + prefix + ' %.2f%%\t' % (now / total * 100), end='')
    print('[' + '>' * int(now / total * length) + '-' * int(length - now / total * length) + ']', end='')
