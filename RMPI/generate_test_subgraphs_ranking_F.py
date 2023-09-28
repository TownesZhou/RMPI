import os
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl
from dgl.data.utils import save_graphs
import torch.nn as nn
def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}

    if saved_relation2id is None:
        relation2id = {}
        rel = 0
    else:
        relation2id = saved_relation2id
        rel = len(saved_relation2id.keys())

    triplets = {}

    ent = 0
    # rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_ = model, adj_list, dgl_adj_list, id2entity, params


def get_neg_samples_replacing_head_tail_rel(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        # neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        # Also sample negative relations
        neg_triplet = {'head': [[], 0], 'tail': [[], 0], 'rel': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])
        
        # Sample negative relations
        neg_triplet['rel'][0].append([head, tail, rel])
        while len(neg_triplet['rel'][0]) < num_samples:
            neg_head = head
            neg_tail = tail
            neg_rel = np.random.choice(r)

            if neg_rel != rel and adj_list[neg_rel][neg_head, neg_tail] == 0:
                neg_triplet['rel'][0].append([neg_head, neg_tail, neg_rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])
        neg_triplet['rel'][0] = np.array(neg_triplet['rel'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


# def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
#     with open(ruleN_pred_path) as f:
#         pred_data = [line.split() for line in f.read().split('\n')[:-1]]
#
#     neg_triplets = []
#     for i in range(len(pred_data) // 3):
#         neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
#         if pred_data[3 * i][1] in saved_relation2id:
#             head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
#             for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
#                 neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
#                 if entity2id[new_head] == head:
#                     neg_triplet['head'][1] = j
#             for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
#                 neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
#                 if entity2id[new_tail] == tail:
#                     neg_triplet['tail'][1] = j
#
#             neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
#             neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])
#
#             neg_triplets.append(neg_triplet)
#
#     return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    # if enclosing_sub_graph:
    enclosing_subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    disclosing_subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    enclosing_subgraph = [adj[enclosing_subgraph_nodes, :][:, enclosing_subgraph_nodes] for adj in A_list]
    disclosing_subgraph = [adj[disclosing_subgraph_nodes, :][:, disclosing_subgraph_nodes] for adj in A_list]

    # labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    enclosing_labels, enclosing_subgraph_nodes_labeled = node_label_new(incidence_matrix(enclosing_subgraph),
                                                                        max_distance=h, enclosing_flag=True)
    disclosing_labels, disclosing_subgraph_nodes_labeled = node_label_new(incidence_matrix(disclosing_subgraph),
                                                                          max_distance=h, enclosing_flag=False)

    pruned_enclosing_subgraph_nodes = np.array(enclosing_subgraph_nodes)[enclosing_subgraph_nodes_labeled].tolist()
    pruned_enclosing_labels = enclosing_labels[enclosing_subgraph_nodes_labeled]

    pruned_disclosing_subgraph_nodes = np.array(disclosing_subgraph_nodes)[disclosing_subgraph_nodes_labeled].tolist()
    pruned_disclosing_labels = disclosing_labels[disclosing_subgraph_nodes_labeled]

    if max_node_label_value is not None:
        pruned_enclosing_labels = np.array(
            [np.minimum(label, max_node_label_value).tolist() for label in pruned_enclosing_labels])
        pruned_disclosing_labels = np.array(
            [np.minimum(label, max_node_label_value).tolist() for label in pruned_disclosing_labels])

    return pruned_enclosing_subgraph_nodes, pruned_enclosing_labels, pruned_disclosing_subgraph_nodes, pruned_disclosing_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1, enclosing_flag=False):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    # enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]

    if enclosing_flag:

        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    else:
        # enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) < 1e6)[0]
        # process the unconnected node (neg samples)
        indices_dim0, indices_dim1 = np.where(labels == 1e7)

        indices_dim1_convert = indices_dim1 + 1
        indices_dim1_convert[indices_dim1_convert == 2] = 0
        new_indices = [indices_dim0.tolist(), indices_dim1_convert.tolist()]
        ori_indices = [indices_dim0.tolist(), indices_dim1.tolist()]

        # values = labels[new_indices] + 1
        # labels[ori_indices] = values
        values = labels[tuple(new_indices)] + 1
        labels[tuple(ori_indices)] = values
        # process the unconnected node (neg samples)

        # print(labels)
        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]

    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph

def prepare_subgraph(dgl_adj_list, nodes, rel, node_labels, max_node_label_value):
    subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
    subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
    subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

    edges_btw_roots = subgraph.edge_id(0, 1)
    rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

    # if rel_link.squeeze().nelement() == 0:
    #     # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
    #     subgraph.add_edge(0, 1)
    #     subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
    #     subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

    if rel_link.squeeze().nelement() == 0:
        subgraph.add_edge(0, 1)
        subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
        subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)
        e_ids = np.zeros(subgraph.number_of_edges())
        e_ids[-1] = 1  # target edge
    else:
        e_ids = np.zeros(subgraph.number_of_edges())
        e_ids[edges_btw_roots] = 1  # target edge

    subgraph.edata['id'] = torch.FloatTensor(e_ids)

    subgraph = prepare_features(subgraph, node_labels, max_node_label_value)
    return subgraph
def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    en_subgraphs = []
    dis_subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        en_nodes, en_node_labels, dis_nodes, dis_node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params.enclosing_sub_graph, max_node_label_value=max_node_label_value)

        en_subgraph = prepare_subgraph(dgl_adj_list, en_nodes, rel, en_node_labels, max_node_label_value)
        dis_subgraph = prepare_subgraph(dgl_adj_list, dis_nodes, rel, dis_node_labels, max_node_label_value)


        en_subgraphs.append(en_subgraph)
        dis_subgraphs.append(dis_subgraph)
        r_labels.append(rel)

    batched_en_graph = dgl.batch(en_subgraphs)
    batched_dis_graph = dgl.batch(dis_subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_en_graph, batched_dis_graph, r_labels)


# def get_rank(neg_links):
#     head_neg_links = neg_links['head'][0]
#     head_target_id = neg_links['head'][1]

#     if head_target_id != 10000:
#         data = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, params.max_label_value)
#         head_scores = model_(data).squeeze(1).detach().numpy()
#         head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
#     else:
#         head_scores = np.array([])
#         head_rank = 10000

#     tail_neg_links = neg_links['tail'][0]
#     tail_target_id = neg_links['tail'][1]

#     if tail_target_id != 10000:
#         data = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, params.max_label_value)
#         tail_scores = model_(data).squeeze(1).detach().numpy()
#         tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
#     else:
#         tail_scores = np.array([])
#         tail_rank = 10000

#     return head_scores, head_rank, tail_scores, tail_rank

def save_subgraphs_to_file(args):
    """
    Compute the subgraphs and save to file for further use.
    """
    run_id, neg_link_id, neg_links = args

    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data_head = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, params.max_label_value)

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data_tail = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, params.max_label_value)

    # Also generate negative relation data
    rel_neg_links = neg_links['rel'][0]
    rel_target_id = neg_links['rel'][1]

    data_rel = get_subgraphs(rel_neg_links, adj_list_, dgl_adj_list_, params.max_label_value)

    # data_head, data_tail, and data_rel are all 3 tuples, each containing (batched_en_graph, batched_dis_graph, r_labels)
    # Both batched_en_graph and batched_dis_graph are dgl batched graphs, each containing len(head_neg_links) respectively.
    # So, we use save_graphs() method to save the data to following 3 files under the "neg_subgraphs/" directory of the test dataset.
    #   data_head -> "run_{run_id}_link_{neg_link_id}_head.bin"
    #   data_tail -> "run_{run_id}_link_{neg_link_id}_tail.bin"
    #   data_rel  -> "run_{run_id}_link_{neg_link_id}_rel.bin"
    # The first two elements of data needs to be unbatched and concatenated (as lists) before saving to file.

    path_to_save = os.path.join('data', params.dataset, 'neg_subgraphs')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    if head_target_id != 10000:
        data_head_0 = dgl.unbatch(data_head[0])
        data_head_1 = dgl.unbatch(data_head[1])
        save_graphs(os.path.join(path_to_save, f'run_{run_id}_link_{neg_link_id}_head.bin'), 
                    data_head_0 + data_head_1,
                    {'labels': data_head[2]})
    
    if tail_target_id != 10000:
        data_tail_0 = dgl.unbatch(data_tail[0])
        data_tail_1 = dgl.unbatch(data_tail[1])
        save_graphs(os.path.join(path_to_save, f'run_{run_id}_link_{neg_link_id}_tail.bin'), 
                    data_tail_0 + data_tail_1,
                    {'labels': data_tail[2]})

    data_rel_0 = dgl.unbatch(data_rel[0])
    data_rel_1 = dgl.unbatch(data_rel[1])
    save_graphs(os.path.join(path_to_save, f'run_{run_id}_link_{neg_link_id}_rel.bin'),
                data_rel_0 + data_rel_1,
                {'labels': data_rel[2]})

    return run_id, neg_link_id


def save_negative_triples_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')






def main(params):
    model = torch.load(params.model_path, map_location='cpu')
    model.params.gpu = -1

    model.device = torch.device('cpu')
    params.max_label_value = np.array([2, 2])

    ori_rel_nums = len(model.relation2id.keys())
    copyed_rel_embed = model.rel_emb.weight.clone()


    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, model.relation2id, add_traspose_rels=False)

    # Split target links into n evenly sized chunks, and take the i-th chunk according to params
    # Note that i starts from 1, and the last chunk may be larger than the rest
    chunk_size = len(triplets['links']) // params.chunk_split[1]
    start_idx = (params.chunk_split[0] - 1) * chunk_size
    end_idx = start_idx + chunk_size if params.chunk_split[0] != params.chunk_split[1] else len(triplets['links'])
    link_chunk = triplets['links'][start_idx:end_idx]

    new_rel_nums = len(relation2id.keys())

    for r in range(1, params.runs+1):
        print(ori_rel_nums)
        print(new_rel_nums)

        added_rel_emb = nn.Embedding(new_rel_nums, 32, sparse=False)
        torch.nn.init.normal_(added_rel_emb.weight)

        for i in range(0, ori_rel_nums):
            added_rel_emb.weight[i] = copyed_rel_embed[i]
        #
        model.rel_emb.weight.data = added_rel_emb.weight.data

        # if params.mode == 'sample':
        #     neg_triplets = get_neg_samples_replacing_head_tail_rel(link_chunk, adj_list)
        #     save_negative_triples_to_file(neg_triplets, id2entity, id2relation)
        # elif params.mode == 'all':
        #     neg_triplets = get_neg_samples_replacing_head_tail_all(link_chunk, adj_list)

        assert params.mode == 'sample', "Only sample mode is supported for now."

        neg_triplets = get_neg_samples_replacing_head_tail_rel(link_chunk, adj_list)
        save_negative_triples_to_file(neg_triplets, id2entity, id2relation)

        func_args = [(r, i, neg_triplet) for i, neg_triplet in enumerate(neg_triplets)]

        with mp.Pool(processes=params.num_workers, initializer=intialize_worker, initargs=(None, adj_list, dgl_adj_list, id2entity, params)) as p:
            for run_id, neg_link_id in tqdm(p.imap_unordered(save_subgraphs_to_file, func_args), total=len(func_args)):
                print(f"Run {run_id} Link {neg_link_id} subgraphs saved to file.")


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--expri_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2", help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--test_file", "-tf", type=str, default="test", help="Name of file containing test triplets")
    parser.add_argument("--num-workers", "-nw", type=int, default=8, help="Number of processes to spawn to save subgraphs")
    parser.add_argument("--chunk-split", "-cs", type=str, default="1/1", 
                        help="Which chunk of target triplets to compute subgraphs for. \
                              i/n means the i-th chunk (start from 1) of a total of n roughly evenly sized chunks. Default: 1/1")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2, help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--mapping', action='store_true', default=False, help='mapping')
    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument("--runs", type=int, default=5, help="How many runs to perform for mean and std?")
    parser.add_argument('--target2nei_atten', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--conc', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--ablation', type=int, default=0, help='0,1 correspond to base, NE')


    params = parser.parse_args()

    params.file_paths = {
        'graph': os.path.join('data', params.dataset, 'train.txt'),
        'links': os.path.join('data', params.dataset, params.test_file+'.txt')
    }

    params.model_path = os.path.join('RMPI/expri_save_models', params.expri_name, 'best_graph_classifier.pth')

    # Split chunks
    chunk_split = params.chunk_split.split('/')
    params.chunk_split = (int(chunk_split[0]), int(chunk_split[1]))  # i-th chunk of a total of n evenly sized chunks

    print('============ Params ============')
    print('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    print('============================================')

    main(params)
