import os
import argparse
import logging
import torch
import random
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import generate_subgraph_datasets

from warnings import simplefilter


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'../data/{params.dataset}/subgraphs_{params.model}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--dataset", "-d", type=str, default="toy", help="Dataset string")
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid", help="Name of file containing validation triplets")

    params = parser.parse_args()

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }
    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic = True
    else:
        params.device = torch.device('cpu')

    main(params)
