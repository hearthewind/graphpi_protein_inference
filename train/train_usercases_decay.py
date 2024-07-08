import configs
# configs.TEST_DATA = ["humanDC", "humanEKC", 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# TEST_DATA = ["humanDC", "humanEKC", 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
configs.TEST_DATA =  ['iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
TEST_DATA = ['iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
from train.util import get_device, build_optimizer, load_torch_algo
from datasets.util import get_proteins_by_fdr, get_proteins_by_fdr_forward
import torch
import numpy as np
import argparse
from train.util import get_dataset, get_model, get_node_features, iprg_converter
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
import os
import pickle

#TEST_DATA = ['iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast", "humanDC"]  # , "yeast", "ups2", "humanDC"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--node_hidden_dim', type=int, default=128)
    parser.add_argument('--num_gnn_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--prior', type=bool, default = False)
    parser.add_argument('--loss_type', type=str, default="cross_entropy")
    parser.add_argument('--pretrain_data_name', type=str, default="epifany")
    parser.add_argument('--filter_psm', type=bool, default=True)
    parser.add_argument('--prior_offset', type=float, default = 0.9)

    args = parser.parse_args()
    print(args)
    return args

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_datasets = []
    for test_data in TEST_DATA:
        test_datasets.append(get_dataset(test_data, train=False, prior=args.prior, filter_psm=args.filter_psm, pretrain_data_name=args.pretrain_data_name))

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
    models = load_torch_algo(f"decay_small_noniso_{args.loss_type}_{args.pretrain_data_name}_prior-{args.prior}_offset-{args.prior_offset}", models, device)
    #models = load_torch_algo(f"original_pretrain_small_noniso_{args.loss_type}_{args.pretrain_data_name}_prior-{args.prior}_offset-{args.prior_offset}", models)

    models = [model.to(device) for model in models]
    models = [model.eval() for model in models]

    len_true_proteins, len_true_proteins_single, len_true_proteins_decoy, result_dict = evaluate(models, test_datasets, device)

    for dataset, data in result_dict.items():
        if "2016" in dataset:
            dataset_prefix, dataset_type = dataset.split("_")
            data_path = os.path.join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_prefix.lower()}", f"mymodel_result",
                                     f"result_{dataset_type}.pkl")
        else:
            data_path = os.path.join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset}", "mymodel_result", "result.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(data, f)

    print(f'optimal true proteins: {len_true_proteins}')
    print(f'optimal true proteins (single): {len_true_proteins_single}')
    print(f'optimal true proteins (decoy): {len_true_proteins_decoy}')

        # else:
        #     print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
        #           f'Val score: {valid_score:.4f}.')


@torch.no_grad()
def evaluate(models, datasets, device):
    print(f'Evaluating ...')

    for model in models:
        model.eval()

    len_true_proteins = {}
    len_true_proteins_single = {}
    len_true_proteins_decoy = {}

    result_dict = {}
    for dataset in datasets:
        edge_index = dataset.edge.T.to(device)
        edge_attr = dataset.edge_weights.to(device)
        test_ids = torch.LongTensor(dataset.proteins)

        x = get_node_features(dataset, models, device)
        out = models[0](x, edge_index, edge_attr)
        protein_scores = out.view(-1)[test_ids].cpu().numpy()
        protein_ids = test_ids.cpu().numpy()
        reverse_id_map = {value:key for key,value in dataset.id_map.items()}
        result_ = {reverse_id_map[protein_id]:score for protein_id, score in zip(protein_ids, protein_scores)}
        result_dict[dataset.dataset] = result_
        result = {iprg_converter(protein, dataset): score for protein, score in result_.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                            for protein_ in protein_pairs] for protein, protein_pairs in dataset.indistinguishable_groups.items()}

        #result =  adjust_scores_by_indistinguishable_groups(result, indishtinguishable_group)
        true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=indishtinguishable_group)
        true_proteins_ = get_proteins_by_fdr_forward(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=indishtinguishable_group)#, indishtinguishable_group=None)
        true_proteins_decoy = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=[], indishtinguishable_group=indishtinguishable_group)#, indishtinguishable_group=None)

        #psuedo_true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=None)

        len_true_proteins_single[dataset.dataset] = len(true_proteins_)
        len_true_proteins[dataset.dataset] = len(true_proteins)
        len_true_proteins_decoy[dataset.dataset] = len(true_proteins_decoy)


    #psuedo_true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=None)
    return len_true_proteins, len_true_proteins_single, len_true_proteins_decoy, result_dict#, result


def adjust_scores_by_indistinguishable_groups(score_dict, indishtinguishable_group):
    for protein, score in score_dict.items():
        if protein in indishtinguishable_group:
            other_indis_proteins = indishtinguishable_group[protein]
            score_dict[protein] = max(score, *[score_dict[indis_protein] for indis_protein in other_indis_proteins])
    return score_dict


# for dataset, data in result_dict.items():
#     if "2016" in dataset:
#         dataset_prefix, dataset_type = dataset.split("_")
#         data_path = os.path.join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_prefix.lower()}", f"mymodel_result",
#                                  f"result_{dataset_type}.pkl")
#     else:
#         data_path = os.path.join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset}", "mymodel_result", "result.pkl")
#
#     with open(data_path, "wb") as f:
#         pickle.dump(data, f)


if __name__ == "__main__":
    main()