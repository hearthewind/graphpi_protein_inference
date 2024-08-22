# configs.TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
# TEST_DATA = ['18mix', 'iPRG2016_B', 'iPRG2016_A', "iPRG2016_AB", "ups2", "yeast"]
from train.train_self import *

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_datasets = []
    for test_data in TEST_DATA:
        test_datasets.append(get_dataset(test_data, train=False, prior=args.prior, prior_offset=args.prior_offset, pretrain_data_name=args.pretrain_data_name))

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

        with open(os.path.join(PROJECT_ROOT_DIR, args.save_result_dir, "groups", f"{group_name}_groups.pkl"), "wb") as f:
            pickle.dump(indishtinguishable_group, f)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    models = get_model(args, device, test_data)

    result_dict, len_true_proteins = store_result(args, test_datasets, device, models)
    num_layers = len(args.model_types.split("_"))
    get_fdr_vs_TP_graphs(save_dir=args.save_result_dir, result_dict=result_dict,
                         image_post_fix=f"{num_layers}_{args.node_hidden_dim}_{args.rounds}_{args.epochs}_{args.lr}")

if __name__ == "__main__":
    main()