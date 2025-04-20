import os.path
import os
import pickle
import yaml
import torch
import argparse
from torch_geometric.data import Data
import random
import numpy as np
from models.product_embedding import Product_Embedding
from models.vit import ViT
from utils.losses import BCEFocalLoss
from trainer.train_model import train
from dataset.MyData import MyDatasetSSL, load_dataset
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--cancer_type', type=str, default='STAD')  # choose pancancer or BRCA or LUAD or PRAD or KIRC or LIHC or LUSC or STAD or THCA
parser.add_argument('--network', type=str, default='CPDB')      # choose BioGRID or CPDB or HINT or IRefIndex or STRING
parser.add_argument('--input_base_path', type=str, default='./data', help='input processed file')
parser.add_argument('--output_base_path', type=str, default='./output', help='output files base path')

parser.add_argument('--dataset_file_path', type=str, default='dataset.pkl', help='input processed file')
parser.add_argument('--ped_dataset_file_path', type=str, default='ped_result.pkl', help='input product embedding file with three channel')

parser.add_argument('--model_train_config_path', type=str, default='./config/train_config.yaml', help='model train_test config file')
parser.add_argument('--model_config_path', type=str, default='./config/model_config.yaml', help='model config file')
parser.add_argument('--model_ped_config_path', type=str, default='./config/ped_config.yaml', help='product_embedding config file')
parser.add_argument('--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--seed', type=int, default=42, help='seed')
args = parser.parse_args()


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def getname(names):
    first_name = ''
    results = []
    for i in range(len(names)):
        name = str(names[i]).split(':')[-1].strip()
        if name == first_name:
            break
        if i == 0:
            first_name = name
        results.append(name)
    return results

def get_cancertype_dict(feature_names):
    result = {}
    count = len(feature_names)
    cancer_names = getname(feature_names)
    cancer_count = len(cancer_names)
    for i in range(count):
        name = cancer_names[i % cancer_count]
        values = result.get(name, [])
        values.append(i)
        result[name] = values
    return result

def main():

    # set seed
    setup_seed(args)

    # dataset path
    args.dataset_file_path = os.path.join(args.input_base_path, args.network, args.dataset_file_path)
    # pyg.dataset
    dataset = load_pkl(args.dataset_file_path)
    data = Data(x=dataset['feature'], y=dataset['label_pos'], edge_index=dataset['edge_index'], mask=dataset['mask_all'],
                node_names=dataset['node_names'])

    # pan cancer or specific_cancer
    if args.cancer_type == 'pancancer':
        data.x = data.x

        args.ped_dataset_file_path = os.path.join(args.input_base_path, args.network, args.ped_dataset_file_path)
        args.output_base_path = os.path.join(args.output_base_path, args.cancer_type, args.network)
    else:
        cancertype_dict = get_cancertype_dict(dataset['feature_names'])
        data.x = data.x[:, cancertype_dict[args.cancer_type]]

        args.ped_dataset_file_path = args.cancer_type + '_' + args.ped_dataset_file_path
        args.ped_dataset_file_path = os.path.join(args.input_base_path, args.network, args.ped_dataset_file_path)

        args.output_base_path = os.path.join(args.output_base_path, 'specific_cancer', args.cancer_type, args.network)
        assert os.path.isdir(args.output_base_path)
    print('Cancer type: ', args.cancer_type)

    # Config files
    train_config = load_config(args.model_train_config_path)
    model_config = load_config(args.model_config_path)
    ped_config = load_config(args.model_ped_config_path)

    # product embedding operation
    if os.path.exists(args.ped_dataset_file_path):
        ped_result = load_pkl(args.ped_dataset_file_path)
    else:
        print("Generate new ped_result.pkl file...")
        in_dim = data.x.shape[1]
        ped = Product_Embedding(ped_config, in_dim)
        ped_result = ped(data)
        with open(args.ped_dataset_file_path, 'wb') as f:
            pickle.dump(ped_result, f, pickle.HIGHEST_PROTOCOL)

    # Get customized dataset
    dataset_mydata = MyDatasetSSL(data, ped_result)

    # DataLoader
    train_loader, test_loader, train_set, train_set_num, train_pos_num = load_dataset(args, dataset_mydata, test_percent=train_config['test_percent'], batch_size=train_config['batch_size'])

    # Model
    model = ViT(model_config).to(args.device)
    if torch.cuda.is_available():
        model = model.to(args.device)

    # Loss function
    # Focal Loss
    alpha = (train_set_num - train_pos_num) / train_set_num  # negative samples number / trainset samples number
    loss_func = BCEFocalLoss(train_config['gamma'], alpha)
    print('alpha of focal loss:', alpha)

    # Cross Entropy loss
    # loss_func = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

    # Train
    train(args, train_config, train_set, test_loader, model, loss_func, optimizer, threshold=train_config['threshold'])
    print('Cancer type: {} network: {} done!!'.format(args.cancer_type, args.network))

if __name__ == '__main__':
    main()
