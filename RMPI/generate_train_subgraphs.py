import os
import shutil
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

    # If db_path already exists, delete the directory and all its contents
    if os.path.isdir(params.db_path):
        shutil.rmtree(params.db_path)

    generate_subgraph_datasets(params)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    # Experiment setup params
    parser.add_argument("--model", type=str, default="RMPI", help="model name")
    parser.add_argument("--expri_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="toy", help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid", help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=20, help="Learning rate of the optimizer")
    parser.add_argument("--eval_every_iter", type=int, default=455, help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=1, help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-2, help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1, help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloading processes")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=False, help='whether to only consider enclosing subgraph')
    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32, help="Relation embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--target2nei_atten', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--conc', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--epoch', type=int, default=0, help='to record epoch')
    parser.add_argument('--ablation', type=int, default=0,
                        help='0,1 correspond to base, NE')

    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')

    params = parser.parse_args()

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
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
