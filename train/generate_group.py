import torch
import numpy as np
import argparse

import configs
from configs import DATA_LIST, TEST_DATA
from train.util import save_torch_algo, get_sampler_hetero as get_sampler
from datasets.util import get_proteins_by_fdr
from train.util import iprg_converter, get_node_features, get_dataset
import random
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR, TEST_DATA
import os
import pickle

configs.TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast", "iPRG2016TS_A", "iPRG2016TS_B", "iPRG2016TS_AB"]
TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast", "iPRG2016TS_A", "iPRG2016TS_B", "iPRG2016TS_AB"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=500)
    parser.add_argument('--opt_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--node_hidden_dim', type=int, default=128)
    parser.add_argument('--num_gnn_layers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--prior', type=bool, default = True)
    parser.add_argument('--prior_offset', type=float, default = 0.9)
    parser.add_argument('--loss_type', type=str, default="cross_entropy")
    parser.add_argument('--pretrain_data_name', type=str, default="epifany")
    parser.add_argument('--protein_label_type', type=str, default="benchmark")


    # For hetero GNN
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 0 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='add',)
    parser.add_argument('--edge_hidden_dim', type=int, default=128)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_deeppep', type=bool, default=False)

    # for self training
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--average', type=bool, default=True)
    parser.add_argument('--concat_old', type=bool, default=False)

    parser.add_argument('--save_result_dir', type=str, default="self_avg")

    args = parser.parse_args()
    print(args)
    return args

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    test_datasets = []
    test_samplers = []
    for test_data in TEST_DATA:
        test_datasets.append(get_dataset(test_data, train=False, prior=args.prior, prior_offset=args.prior_offset, pretrain_data_name=args.pretrain_data_name, use_deeppep=args.use_deeppep))
        if test_data != "yeast":
            if args.batch_size <= 0:
                test_samplers.append(test_datasets[-1])
            else:
                test_sampler = get_sampler(test_datasets[-1], test_datasets[-1].proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers, rebalance=False)
                test_samplers.append((test_datasets[-1], test_sampler))

    for test_data in test_datasets:
        reverse_id_map = {value: key for key, value in test_data.id_map.items()}
        indishtinguishable_group = {reverse_id_map[protein]: [reverse_id_map[protein_] \
                                                              for protein_ in protein_pairs] for protein, protein_pairs
                                    in test_data.indistinguishable_groups.items()}

        if test_data.dataset == 'iPRG2016_A':
            group_name = 'iprg_a'
        elif test_data.dataset == 'iPRG2016_B':
            group_name = 'iprg_b'
        elif test_data.dataset == 'iPRG2016_AB':
            group_name = 'iprg_ab'
        elif test_data.dataset == 'iPRG2016TS_A':
            group_name = 'iprg_a_ts'
        elif test_data.dataset == 'iPRG2016TS_B':
            group_name = 'iprg_b_ts'
        elif test_data.dataset == 'iPRG2016TS_AB':
            group_name = 'iprg_ab_ts'
        else:
            group_name = test_data.dataset

        with open(os.path.join(PROJECT_ROOT_DIR, "outputs", f"{group_name}_groups.pkl"), "wb") as f:
            pickle.dump(indishtinguishable_group, f)

        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.01, indishtinguishable_group=indishtinguishable_group)))

    valid_datasets = [dataset for dataset in test_datasets if dataset.dataset != "yeast"]

    return

if __name__ == "__main__":
    main()