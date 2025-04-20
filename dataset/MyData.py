
import torch
from torch.utils.data import random_split
import pickle
import argparse
from torch_geometric.data import Data
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

# Customizing the dataset class and dataloader approach

def load_dataset(args, dataset_mydata, test_percent, batch_size, shuffle=False, semi=False):
    """
    :param dataset_mydata: MyDatasetSSL Class
    :param test_percent: Percentage of test set
    :param batch_size:
    :param shuffle: Default disorder
    :param semi: Whether semi-supervised or not, the default is fully supervised training, which only reads labeled data
    :return:
    """
    if semi:
        train_set, test_set = split_train_test(args, test_percent, dataset=dataset_mydata)
        # DataLoader
        train_loader = DataLoader(train_set, batch_size, shuffle)
        test_loader = DataLoader(test_set, batch_size, shuffle)
    else:
        # To read only labeled data, you need to use the Subset method to get the dataset_mydata_l
        labeled_all_idx = torch.nonzero(torch.tensor(dataset_mydata.mask_all))
        # labeled_pos_idx = torch.nonzero(torch.tensor(dataset_mydata.y_pos))
        # labeled_neg_idx = torch.nonzero(np.logical_xor(torch.tensor(dataset_mydata.mask_all), torch.tensor(dataset_mydata.y_pos)))

        dataset_mydata_l = Subset(dataset_mydata, labeled_all_idx)

        train_set, test_set = split_train_test(args, test_percent, dataset=dataset_mydata_l)

        # On the training set: extract the data corresponding to train_set from dataset_mydata
        train_set_data = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][0] for idx in train_set.indices]
        train_set_y_pos = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][1] for idx in train_set.indices]
        train_set_mask_all = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][2] for idx in train_set.indices]
        train_set_gene_name_all = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][3] for idx in train_set.indices]

        # On the test set: extract the data corresponding to test_set from dataset_mydata
        test_set_data = [test_set.dataset.dataset[int(np.array(test_set.dataset.indices[idx], np.int32))][0] for idx in test_set.indices]
        test_set_y_pos = [test_set.dataset.dataset[int(np.array(test_set.dataset.indices[idx], np.int32))][1] for idx in test_set.indices]
        test_set_mask_all = [test_set.dataset.dataset[int(np.array(test_set.dataset.indices[idx], np.int32))][2] for idx in test_set.indices]
        test_set_gene_name_all = [test_set.dataset.dataset[int(np.array(test_set.dataset.indices[idx], np.int32))][3] for idx in test_set.indices]


        # Packaging for data pairs
        sampled_train_set = [(X, y, mask, gn) for X, y, mask, gn in zip(train_set_data, train_set_y_pos, train_set_mask_all, train_set_gene_name_all)]
        sampled_test_set = [(X, y, mask, gn) for X, y, mask, gn in zip(test_set_data, test_set_y_pos, test_set_mask_all, test_set_gene_name_all)]

        # dataloader
        train_loader = DataLoader(sampled_train_set, batch_size, shuffle)
        test_loader = DataLoader(sampled_test_set, batch_size, shuffle)

    return train_loader, test_loader, train_set, sum(train_set_mask_all), int(np.array(train_set_y_pos).sum())


def get_mask(args, dataset_mydata, test_percent):
    # To read only labeled data, you need to use the Subset method to get the dataset_mydata_l
    labeled_all_idx = torch.nonzero(torch.tensor(dataset_mydata.mask_all))
    # labeled_pos_idx = torch.nonzero(torch.tensor(dataset_mydata.y_pos))
    # labeled_neg_idx = torch.nonzero(np.logical_xor(torch.tensor(dataset_mydata.mask_all), torch.tensor(dataset_mydata.y_pos)))

    dataset_mydata_l = Subset(dataset_mydata, labeled_all_idx)

    train_set, test_set = split_train_test(args, test_percent, dataset=dataset_mydata_l)

    train_index = [np.array(train_set.dataset.indices[idx], np.int32) for idx in train_set.indices]
    test_index = [np.array(test_set.dataset.indices[idx], np.int32) for idx in test_set.indices]

    train_mask = np.zeros(len(dataset_mydata.mask_all), )
    for i in range(len(train_index)):
        train_mask[train_index[i]] = 1

    te_mask = np.zeros(len(dataset_mydata.mask_all), )
    for i in range(len(test_index)):
        te_mask[test_index[i]] = 1

    return train_mask.astype(bool), te_mask.astype(bool)


# Divide the dataset: 80% training set, 20% test set
def split_train_test(args, test_percent, dataset):
    torch.manual_seed(args.seed)
    num_samples = len(dataset)
    test_num = int(test_percent * num_samples)
    train_num = num_samples - test_num
    train_set, test_set = random_split(dataset, [train_num, test_num])
    return train_set, test_set

class MyDatasetSSL(Dataset):
    def __init__(self, data, X):
        """
        :param data: data of PYG
        :param X: ped_result
        """
        # Read data
        self.x_data = X
        self.y_pos = data['y']
        self.mask_all = data['mask']
        self.gene_names = data['node_names']

    def __getitem__(self, index):
        # Returns data and corresponding tags based on index
        return self.x_data[index], self.y_pos[index], self.mask_all[index], self.gene_names[index]

    def __len__(self):
        # Returns the number of data
        return len(self.x_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()


    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # input file
    dataset = load_pkl('../data/HINT/dataset.pkl')
    data = Data(x=dataset['feature'], y=dataset['label_pos'], edge_index=dataset['edge_index'], mask=dataset['mask_all'],
                node_names=dataset['node_names'])
    ped_dataset = load_pkl('../data/HINT/ped_result_ch1.pkl')

    # Get customized dataset
    dataset_mydata = MyDatasetSSL(data, ped_dataset)

    # DataLoader
    train_loader, test_loader, train_set, train_set_num, train_pos_num = load_dataset(args, dataset_mydata, test_percent=0.2, batch_size=4)

    # Example of iterating a data set
    for batch_idx, data in enumerate(train_loader):
        """data: values of dataset
           label: positive labels of dataset
           mask_all : labeled samples
           gene_name: node name
        """
        X, y_pos, mask_all, gene_name = data
        print(batch_idx)
        print(data)




