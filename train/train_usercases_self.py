import configs
# configs.TEST_DATA = ["humanDC", "humanEKC", 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# TEST_DATA = ["humanDC", "humanEKC", 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# configs.TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# configs.TEST_DATA = ['18mix']
# TEST_DATA = ['18mix']

configs.TEST_DATA = ['hela_3t3']
TEST_DATA = ['hela_3t3']

from train.util import get_device, build_optimizer, load_torch_algo, get_dataset, get_node_features, iprg_converter
from datasets.util import get_proteins_by_fdr, get_proteins_by_fdr_forward
from train.util import get_model_hetero as get_model
import torch
import numpy as np
import argparse
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
import os
import pickle
from train.train_self import *

#TEST_DATA = ['iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast", "humanDC"]  # , "yeast", "ups2", "humanDC"]

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_datasets = []
    for test_data in TEST_DATA:
        test_datasets.append(get_dataset(test_data, train=False, prior=args.prior, prior_offset=args.prior_offset, pretrain_data_name=args.pretrain_data_name))

    #protein_scores = adjust_scores_by_indistinguishable_groups(test_data.protein_scores, indishtinguishable_group)
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
        else:
            group_name = test_data.dataset

        with open(os.path.join(PROJECT_ROOT_DIR, "outputs", f"{group_name}_groups.pkl"), "wb") as f:
            pickle.dump(indishtinguishable_group, f)

        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=indishtinguishable_group)))
        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=None)))
        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=[], fdr=0.05, indishtinguishable_group=indishtinguishable_group)))
        #test_proteins = torch.LongTensor(test_data.proteins)

    device = get_device()

    models = get_model(args, device, test_data)

    result_dict, len_true_proteins = store_result(args, test_datasets, device, models)
    num_layers = len(args.model_types.split("_"))
    get_fdr_vs_TP_graphs(save_dir=args.save_result_dir, result_dict=result_dict,
                         image_post_fix=f"{num_layers}_{args.node_hidden_dim}_{args.rounds}_{args.epochs}_{args.lr}")

if __name__ == "__main__":
    main()