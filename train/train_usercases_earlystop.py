import configs
# configs.TEST_DATA = ["humanDC", "humanEKC", 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# TEST_DATA = ["humanDC", "humanEKC", 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
configs.TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
from train.util import get_device, build_optimizer, load_torch_algo, get_dataset, get_node_features, iprg_converter
from datasets.util import get_proteins_by_fdr, get_proteins_by_fdr_forward
from train.util import get_model_hetero as get_model
import torch
import numpy as np
import argparse
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
import os
import pickle
from train.train_semi_earlystop import *

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

        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=indishtinguishable_group)))
        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=None)))
        print(f"benchmark score ({args.pretrain_data_name}) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=[], fdr=0.05, indishtinguishable_group=indishtinguishable_group)))
        #test_proteins = torch.LongTensor(test_data.proteins)

    device = get_device()

    models = get_model(args, device, test_data)

    store_results(models, test_datasets, device, args)
    get_fdr_vs_TP_graphs()

if __name__ == "__main__":
    main()