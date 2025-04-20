## Reference the code of HGDC: https://github.com/NWPU-903PR/HGDC.

import pandas as pd
import numpy as np
import torch
import pickle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from itertools import chain

def load_network(file_path):
    """
    Load network from file.
    :param file_path: Full pathname of the network file
    :return: net (class: pandas.DataFrame): Edges in the network, nodes (class: pandas.DataFrame): The nodes in the network
    """
    net = pd.read_csv(file_path, header=0, names=['source', 'target', 'confidence'], sep='\t')
    net = net.iloc[:, 0:2]

    nodes = pd.concat([net['source'], net['target']], ignore_index=True)
    nodes = pd.DataFrame(nodes, columns=['nodes']).drop_duplicates()
    nodes.reset_index(drop=True, inplace=True)
    return net, nodes

def build_customized_feature_matrix(feat_file_lst, network_file, feat_name_lst):
    """
    Build feature matrix on your own data.
    :param feat_file_lst: List of full pathnames of feature files. Each feat_file in feat_file_lst contains two columns, i.e., gene names and feature values.
    :param network_file: Full pathname of network file
    :param feat_name_lst: List of feature names
    :return: Concatenated feature matrix with n rows(genes) and m columns(features) (class: pandas.DataFrame)
    """
    feat_dic = dict()
    # Load gene features from each feat_file
    for i in range(0, len(feat_file_lst)):
        feat_dic[feat_name_lst[i]] = pd.read_csv(feat_file_lst[i], sep='\t', index_col=0)
    # Load network from file
    net, net_nodes = load_network(network_file)
    # Normalization by StandardScaler
    scaler = preprocessing.StandardScaler()
    feat_raw = scaler.fit_transform(np.abs(feat_dic[feat_name_lst[0]].reindex(net_nodes['nodes'].values.tolist(), fill_value=0)))
    # Concatenate multiple features to form one feature matrix
    if len(feat_file_lst) > 1:
        for i in range(1, len(feat_file_lst)):
            feat_raw = np.concatenate((feat_raw, scaler.fit_transform(np.abs(feat_dic[feat_name_lst[i]].reindex(net_nodes['nodes'].values.tolist(), fill_value=0)))), axis=1)


    columns_name = [feat_dic[feat_name_lst[i]].columns for i in range(len(feat_file_lst))]
    full_list = list(chain(columns_name[0], columns_name[1], columns_name[2], columns_name[3]))
    return pd.DataFrame(feat_raw, index=net_nodes['nodes'].values.tolist(), columns=full_list)

def create_edge_index(network_file,net_features):
    """
    Convert the edges in a network into edges indexed by integer ids, which is necessary to build an object typeof torch_geometric.data.Data.
    :param network_file: Full pathname of the network file
    :param net_features (class: pandas.DataFrame): Concatenated feature matrix with n rows(genes) and m columns(features)
    :return (class: pandas.DataFrame): Edges indexed by integer ids
    """
    net, _ = load_network(network_file)
    node_df = pd.DataFrame({'name':net_features.index.values.tolist(),
                            'id':[i for i in np.arange(0,net_features.shape[0])]})
    net = pd.merge(left=net,right=node_df,how='left',left_on='source',right_on='name')
    net.columns=['source','target','sourcename','sourceid']
    net = pd.merge(left=net, right=node_df, how='left',left_on='target',right_on='name')
    net.columns=['source','target','sourcename','sourceid','targetname','targetid']
    edge_index1 = net.loc[:,['sourceid','targetid']]
    # Treat the graph as undirected graph
    edge_index2 = net.loc[:,['targetid','sourceid']]
    edge_index = pd.concat([edge_index1,edge_index2],axis=0)
    return edge_index

feat_file_lst = ['../data/pancancer/copy_number_variation/pancancer_gene_CNACS_matrix.tsv',
                 '../data/pancancer/single_nucleotide_variation/pancancer_gene_mutation_matrix.tsv',
                 '../data/pancancer/single_nucleotide_variation/pancancer_mean_mutation_matrix.tsv',
                 '../data/pancancer/gene_expression/pancancer_mean_expression_fc_matrix.tsv']

network_file = '../data/STRING/PPI_edge_index.tsv'

feat_name_lst = ['cnv_count', 'snv_count', 'snv', 'exp']

# Concatenate multiple features to form one feature matrix
feature_matrix = build_customized_feature_matrix(feat_file_lst, network_file, feat_name_lst)
# save as csv file
# feature_matrix.to_csv(network_file.split("/")[2]+'_feature_matrix.csv', index=True)

# A dataset contains the following data:
# feature: the gene feature matrix
# edge_index: graph edges for training model
# node_name: gene names
# feature_name: feature names
# label_pos: True labels of genes (1 for positive samples),
# mask_all: mask for training a single model without cross-validation
dataset = dict()
dataset['feature'] = torch.FloatTensor(np.array(feature_matrix))
dataset['node_names'] = feature_matrix.index.values.tolist()
# Create edge_index by edges in network file
edge_index = create_edge_index(network_file, feature_matrix)
dataset['edge_index'] = torch.LongTensor(np.array(edge_index).transpose())
dataset['feature_names'] = feature_matrix.columns.values.tolist()

# Canonical driver genes (positive samples)
d_lst = pd.read_table(filepath_or_buffer='../data/865_drivers.txt', sep='\t', header=None, index_col=None,
                      names=['driver'])
d_lst = d_lst['driver'].values.tolist()

# Nondriver genes (negative samples)
nd_lst = pd.read_table(filepath_or_buffer='../data/2187_nondrivers.txt', sep='\t', header=None,
                       index_col=None, names=['nondriver'])
nd_lst = nd_lst['nondriver'].values.tolist()

labels_pos = []
mask_all = []  # mask for training a single model without cross-validation
for g in dataset['node_names']:
    if g in d_lst:
        labels_pos.append(1)
    else:
        labels_pos.append(0)

    if (g in d_lst) or (g in nd_lst):
        mask_all.append(True)
    else:
        mask_all.append(False)

# dataset['label_pos'] = torch.FloatTensor(np.array(labels_pos))
dataset['label_pos'] = np.array(labels_pos)
dataset['mask_all'] = np.array(mask_all)


d_in_net = []  # Canonical driver genes in the network
nd_in_net = []  # Nondriver genes in the network
for g in dataset['node_names']:
    if g in d_lst:
        d_in_net.append(g)
    elif g in nd_lst:
        nd_in_net.append(g)

print('The Number of Cononical Driver Genes in the network:', len(d_in_net))
print('The Number of Nondriver Genes in the network:', len(nd_in_net))



# Save the dataset as pickle file
with open('../data/CPDB/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)