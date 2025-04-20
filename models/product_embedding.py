
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm

# outer product operation
def outer_product(emb1, emb2, dim):
    """
    :param emb1: the embedding representation of bioinfo data1, here is the embedding after three GNN layers
    :param emb2: the embedding representation of bioinfo data2, here is the feature matrix embedding representation
    :param dim: embedding dimension of emb1 and emb2
    :return: result: the matrix with shape: (N,dim,dim) after outer_product operation, N is the node numbers(len(emb1))
    """
    result = []

    for i in range(len(emb1)):
        emb1_emb2_outer = np.dot(emb1[i].reshape(dim, 1), emb2[i].reshape(1, dim))
        result.append(emb1_emb2_outer)

    return np.array(result)  # list-->array

def generate_fake_image(x1, x2, x3):

    result = []
    for i in range(x1.shape[0]):
        fi = np.concatenate((x1[i], x2[i], x3[i]), axis=0)
        fi = fi.reshape(-1, x1.shape[1], x1.shape[1])
        result.append(fi)
    return result

class Product_Embedding(nn.Module):
    def __init__(self, ped_config, in_dim):
        super().__init__()
        self.ped_config = ped_config
        self.in_dim = in_dim
        self.out_dim = ped_config['out_dim']
        self.dropout = ped_config['dropout']
        self.hid_num1 = ped_config['hidden_dim1']
        self.hid_num2 = ped_config['hidden_dim2']
        self.channel = ped_config['channel']

        self.conv1 = GCNConv(self.in_dim, self.hid_num1)
        self.layer_norm1 = LayerNorm(self.hid_num1)
        self.conv2 = GCNConv(self.hid_num1, self.hid_num2)
        self.layer_norm2 = LayerNorm(self.hid_num2)
        self.conv3 = GCNConv(self.hid_num2, self.out_dim)

        self.linear1 = nn.Linear(self.in_dim, self.hid_num1)
        self.linear2 = nn.Linear(self.hid_num1, self.hid_num2)
        self.linear3 = nn.Linear(self.hid_num2, self.out_dim)

    def forward(self, data):
        # x is the original feature matrix value
        x, edge_index = data.x, data.edge_index

        # x_1 is embedding after one layer of GNN
        x_1 = self.conv1(x, edge_index)
        x_11 = F.relu(x_1)
        x_111 = F.dropout(x_11, p=self.dropout, training=self.training)

        # x_2 is embedding after two GNN layers
        x_1111 = self.layer_norm1(x_111)
        x_2 = self.conv2(x_1111, edge_index)
        x_22 = F.relu(x_2)
        x_222 = F.dropout(x_22, p=self.dropout, training=self.training)

        # x_3 is embedding after three GNN layers
        x_2222 = self.layer_norm2(x_222)
        x_3 = self.conv3(x_2222, edge_index)

        x_sc = x_1 + x_3

        # feature matrix
        lx_1 = self.linear1(x)
        lx_11 = F.relu(lx_1)
        lx_111 = F.dropout(lx_11, p=self.dropout, training=self.training)

        lx_2 = self.linear2(lx_111)
        lx_22 = F.relu(lx_2)
        lx_222 = F.dropout(lx_22, p=self.dropout, training=self.training)

        lx_3 = self.linear3(lx_222)

        lx_sc = lx_1 + lx_3

        # combine
        if self.channel == 3:
            ch1 = outer_product(x_1.detach().numpy(), lx_1.detach().numpy(), self.out_dim)
            ch2 = outer_product(x_3.detach().numpy(), lx_3.detach().numpy(), self.out_dim)
            ch3 = outer_product(x_sc.detach().numpy(), lx_sc.detach().numpy(), self.out_dim)
            #
            ped_result = generate_fake_image(ch1, ch2, ch3)
        else:
            # ablation(only one chanel)
            ped_result = outer_product(x_3.detach().numpy(), lx_3.detach().numpy(), self.out_dim)

        return ped_result