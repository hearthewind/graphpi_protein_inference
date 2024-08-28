import gc
from train.util import build_optimizer, load_torch_algo, compute_pauc
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from configs import DATA_LIST, TEST_DATA
from sklearn.metrics import roc_auc_score
from train.util import save_torch_algo, get_sampler_hetero as get_sampler
from datasets.util import get_proteins_by_fdr
from train.util import iprg_converter, get_node_features, get_dataset
from train.util import get_model_hetero as get_model
import random
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
import os
import pickle
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=500)
    parser.add_argument('--opt_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--node_hidden_dim', type=int, default=100)
    parser.add_argument('--num_gnn_layers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--prior', type=bool, default = True)
    parser.add_argument('--prior_offset', type=float, default = 0.9)
    parser.add_argument('--loss_type', type=str, default="cross_entropy")
    parser.add_argument('--pretrain_data_name', type=str, default="epifany")
    parser.add_argument('--protein_label_type', type=str, default="benchmark")


    # For hetero GNN
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE_EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 0 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='add',)
    parser.add_argument('--edge_hidden_dim', type=int, default=100)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_deeppep', type=bool, default=False)

    # for self training
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--average', type=bool, default=True)
    parser.add_argument('--concat_old', type=bool, default=False)
    parser.add_argument('--save_result_dir', type=str, default="results")

    parser.add_argument('-f')

    args = parser.parse_args()
    print(args)
    return args

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    data_samplers = []
    st_samplers = []
    print('train data list:', DATA_LIST)

    for dataset in DATA_LIST:
        if dataset in TEST_DATA:
            data = get_dataset(dataset, train=True, prior=args.prior, prior_offset=args.prior_offset, protein_label_type="decoy", pretrain_data_name=args.pretrain_data_name, loss_type=args.loss_type, use_deeppep=args.use_deeppep)
        else:
            data = get_dataset(dataset, train=True, prior=args.prior, prior_offset=args.prior_offset, protein_label_type=args.protein_label_type, pretrain_data_name=args.pretrain_data_name, loss_type=args.loss_type, use_deeppep=args.use_deeppep)
        num_valid = int(len(data.proteins)*0.2)
        train_node_idx = np.array(data.proteins)
        np.random.shuffle(train_node_idx)
        valid_proteins = torch.LongTensor(train_node_idx[:num_valid])
        train_proteins = torch.LongTensor(train_node_idx[num_valid:])

        if args.batch_size <= 0:
            data_samplers.append((data, train_proteins, valid_proteins))
            st_samplers.append((data, torch.LongTensor(data.proteins)))
        else:
            train_sampler = get_sampler(data, train_proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers, rebalance=False)
            valid_sampler = get_sampler(data, valid_proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers, rebalance=False)
            all_sampler = get_sampler(data, torch.LongTensor(list(data.protein_map.values())), args.batch_size, fan_out=[-1] * args.num_gnn_layers, rebalance=False)
            data_samplers.append((data, train_sampler, valid_sampler))
            st_samplers.append((data, all_sampler))

        gc.collect()

    test_datasets = []
    test_samplers = []
    for test_data in TEST_DATA:
        test_datasets.append(get_dataset(test_data, train=False, prior=args.prior, prior_offset=args.prior_offset, pretrain_data_name=args.pretrain_data_name, use_deeppep=args.use_deeppep))
        if test_data not in  ["yeast", "hela_3t3"]:
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
        else:
            group_name = test_data.dataset

        path = os.path.join(PROJECT_ROOT_DIR, args.save_result_dir, "groups")
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f"{group_name}_groups.pkl"), "wb") as f:
            pickle.dump(indishtinguishable_group, f)

    valid_datasets = [dataset for dataset in test_datasets if dataset.dataset != "yeast"]


    train_pipeline(args, data_samplers, st_samplers, test_datasets, test_samplers, valid_datasets, test_data, device, evaluate_epoch_num=1)


def train_pipeline(args, data_samplers, st_samplers, test_datasets, test_samplers, valid_datasets, test_data, device, evaluate_epoch_num=1):

    models = get_model(args, device, test_data)

    scheduler, optimizer = build_optimizer(args, models.parameters())

    best_valid_score = 0
    best_valid_loss = np.inf

    for epoch in range(args.epochs+1):

        if args.batch_size <= 0:
            train_loss = train_hetero(models, data_samplers, device, optimizer, args.loss_type)
            valid_loss, valid_score = test_hetero(models, data_samplers, device, args.loss_type)

            #if epoch % evaluate_epoch_num == 0:
            #    len_true_proteins, _, pauc_dict = evaluate_hetero(models, valid_datasets, device)
        else:
            train_loss = train_hetero_batch(models, data_samplers, device, optimizer, args.loss_type)
            valid_loss, valid_score = test_hetero_batch(models, data_samplers, device, args.loss_type)
            #if epoch % evaluate_epoch_num == 0:
            #   len_true_proteins, _ = evaluate_hetero_batch(models, test_samplers, device)

        model_name = f"{args.save_result_dir}/trained_model/pretrain_st_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}"
        if args.loss_type == "soft_entropy":
            if best_valid_loss > valid_loss:
                #best_true_proteins = len_true_proteins
                best_valid_loss = valid_loss
                save_torch_algo(models,model_name)
                best_params = copy.deepcopy(models.state_dict())
            if epoch % evaluate_epoch_num == 0:
                # print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}. Num true proteins: {len_true_proteins},\
                #  optimal true proteins: {best_true_proteins}')
                print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}.')
        elif args.loss_type == "cross_entropy":
            if best_valid_score < valid_score:
                #best_true_proteins = len_true_proteins
                #best_pauc = pauc_dict
                best_valid_score = valid_score
                save_torch_algo(models, model_name)
                best_params = copy.deepcopy(models.state_dict())
            if epoch % evaluate_epoch_num == 0:
                # print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
                #       f'Val score: {valid_score:.4f}. Num true proteins: {len_true_proteins}, optimal true proteins: {best_true_proteins}, optimal partial AUC: {best_pauc}')
                print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
                      f'Val score: {valid_score:.4f}. ')

    ### start selftraining
    models, best_params = self_training(args, st_samplers, device, models, best_params, valid_datasets, test_samplers, evaluate_epoch_num)
    models.load_state_dict(best_params)

    #best_true_proteins, _, best_pauc = evaluate_hetero(models, valid_datasets, device)

    result_dict, len_true_proteins = store_result(args, test_datasets, device, models, test_fdr=0.05)
    #num_layers = len(args.model_types.split("_"))
    #get_fdr_vs_TP_graphs(save_dir=args.save_result_dir, result_dict=result_dict, image_post_fix=f"{num_layers}_{args.node_hidden_dim}_{args.rounds}_{args.epochs}_{args.lr}")
    return len_true_proteins


def self_training(args, st_samplers, device, old_models, old_params, valid_datasets, test_samplers, evaluate_epoch_num=1):
    for i in range(1, args.rounds + 1):
        models, best_params = one_round_self_training(args, i, st_samplers, device, old_models, old_params, valid_datasets, test_samplers, evaluate_epoch_num)
        old_models = models
        old_params = best_params

    return models, best_params


def one_round_self_training(args, i, st_samplers, device, old_models, old_params, valid_datasets, test_samplers, evaluate_epoch_num=1):
    st_fdr = 0.05
    old_models.load_state_dict(old_params)

    if args.batch_size <= 0:
        train_datasets = [x[0] for x in st_samplers]
        train_len_true_proteins, train_result_dict, train_pauc_dict = evaluate_hetero(old_models, train_datasets, device)
    else:
        train_len_true_proteins, train_result_dict = evaluate_hetero_batch(old_models, st_samplers, device)

    for dataset, _ in st_samplers:
        result_dict = train_result_dict[dataset.dataset]
        contaminate_proteins = dataset.get_contaminate_proteins()

        reverse_id_map = {value: key for key, value in dataset.protein_map.items()}
        indishtinguishable_group = {
            reverse_id_map[protein]: [reverse_id_map[protein_] for protein_ in protein_pairs] for protein, protein_pairs
            in dataset.indistinguishable_groups.items()}

        true_proteins = get_proteins_by_fdr(result_dict, fdr=st_fdr, contaminate_proteins=contaminate_proteins,
                                            indishtinguishable_group=indishtinguishable_group)
        true_proteins = sorted(true_proteins, key=lambda x: result_dict[x], reverse=True)

        all_proteins = dataset.protein_map.keys()
        false_proteins = set(all_proteins) - set(true_proteins)
        false_proteins = sorted(false_proteins, key=lambda x: result_dict[x] if x in result_dict else 0.0, reverse=True)

        dataset.update_selftrain(args, true_proteins, false_proteins, result_dict)

        print(f'After update, the number of positive proteins is {len(dataset.pos_proteins)}, the number of negative proteins is {len(dataset.neg_proteins)}')

    new_data_samplers = []

    train_datasets = [x[0] for x in st_samplers]
    for data in train_datasets:
        num_valid = int(len(data.proteins) * 0.2)
        train_node_idx = np.array(data.proteins)
        np.random.shuffle(train_node_idx)
        valid_proteins = torch.LongTensor(train_node_idx[:num_valid])
        train_proteins = torch.LongTensor(train_node_idx[num_valid:])

        if args.batch_size <= 0:
            new_data_samplers.append((data, train_proteins, valid_proteins))
        else:
            train_sampler = get_sampler(data, train_proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers,
                                        rebalance=False)
            valid_sampler = get_sampler(data, valid_proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers,
                                        rebalance=False)
            # train_sampler = get_sampler(data, train_proteins, args.batch_size, fan_out=[25, 10, 10, 10], rebalance=False)
            # valid_sampler = get_sampler(data, valid_proteins, args.batch_size, fan_out=[25, 10, 10, 10], rebalance=False)
            new_data_samplers.append((data, train_sampler, valid_sampler))

        gc.collect()

    models = get_model(args, device, valid_datasets[0])
    scheduler, optimizer = build_optimizer(args, models.parameters())

    best_valid_score = 0
    best_valid_loss = np.inf

    model_name = f"{args.save_result_dir}/trained_model/selftrain{i}_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}"
    for epoch in range(args.epochs+1):

        if args.batch_size <= 0:
            train_loss = train_hetero(models, new_data_samplers, device, optimizer, args.loss_type)
            valid_loss, valid_score = test_hetero(models, new_data_samplers, device, args.loss_type)
            # if epoch % evaluate_epoch_num == 0:
            #     len_true_proteins, _, pauc_dict = evaluate_hetero(models, valid_datasets, device)
        else:
            train_loss = train_hetero_batch(models, new_data_samplers, device, optimizer, args.loss_type)
            valid_loss, valid_score = test_hetero_batch(models, new_data_samplers, device, args.loss_type)
            # if epoch % evaluate_epoch_num == 0:
            #     len_true_proteins, _ = evaluate_hetero_batch(models, test_samplers, device)

        if args.loss_type == "soft_entropy":
            if best_valid_loss > valid_loss:
                #best_true_proteins = len_true_proteins
                best_valid_loss = valid_loss
                save_torch_algo(models, model_name)
                best_params = copy.deepcopy(models.state_dict())

            if epoch % evaluate_epoch_num == 0:
                # print(
                #     f'SelfTrain {i} Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}. Num true proteins: {len_true_proteins},\
                #          optimal true proteins: {best_true_proteins}')
                print(
                    f'SelfTrain {i} Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}. ')

        elif args.loss_type == "cross_entropy":
            if best_valid_score < valid_score:
                #best_true_proteins = len_true_proteins
                best_valid_score = valid_score
                #best_pauc = pauc_dict
                save_torch_algo(models, model_name)
                best_params = copy.deepcopy(models.state_dict())
            if epoch % evaluate_epoch_num == 0:
                # print(f'SelfTrain {i} Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
                #       f'Val score: {valid_score:.4f}. Num true proteins: {len_true_proteins}, optimal true proteins: {best_true_proteins}, optimal pauc: {best_pauc}')
                print(f'SelfTrain {i} Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
                      f'Val score: {valid_score:.4f}. ')

    return models, best_params


def store_result(args, test_datasets, device, models, test_fdr=0.05):
    if args.average:
        return store_result_avg(args, test_datasets, device, test_fdr=test_fdr)
    else:
        return store_results_onemodel(models, test_datasets, device)


def store_result_avg(args, test_datasets, device, test_fdr=0.05):
    models = get_model(args, device, test_datasets[0])
    models.eval()
    model_name = f"{args.save_result_dir}/trained_model/pretrain_st_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}"
    models = load_torch_algo(model_name, models, device)

    _, result_dict, _ = evaluate_hetero(models, test_datasets, device)
    final_result_dict = {}
    for dataset_name, data_result in result_dict.items():
        final_result_dict[dataset_name] = {}

        for key, value in data_result.items():
            final_result_dict[dataset_name][key] = [value]

    for i in range(1, args.rounds + 1):
        model_name = f"{args.save_result_dir}/trained_model/selftrain{i}_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}"
        models = load_torch_algo(model_name, models, device)
        _, result_dict, _ = evaluate_hetero(models, test_datasets, device)

        for dataset_name, data_result in result_dict.items():
            for key, value in data_result.items():

                if key not in final_result_dict[dataset_name]:
                    final_result_dict[dataset_name][key] = [value]
                else:
                    final_result_dict[dataset_name][key].append(value)

    avg_result_dict = {}
    for dataset_name, data_result in final_result_dict.items():
        avg_result_dict[dataset_name] = {}
        for key, value in data_result.items():
            avg_result_dict[dataset_name][key] = np.mean(value)

    len_true_proteins = {}
    for dataset in test_datasets:
        result = avg_result_dict[dataset.dataset]
        reverse_id_map = {value: key for key, value in dataset.protein_map.items()}

        result = {iprg_converter(protein, dataset): score for protein, score in result.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {
            iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                                               for protein_ in protein_pairs] for protein, protein_pairs
            in dataset.indistinguishable_groups.items()}

        true_proteins = get_proteins_by_fdr(result, fdr=test_fdr, contaminate_proteins=contaminate_proteins,
                                            indishtinguishable_group=indishtinguishable_group)

        len_true_proteins[dataset.dataset] = len(true_proteins)

    for dataset, data in avg_result_dict.items():
        if "2016" in dataset:
            dataset_prefix, dataset_type = dataset.split("_")
            path = os.path.join(PROJECT_ROOT_DIR, args.save_result_dir, "predictions", f"{dataset_prefix.lower()}", f"graphpi_result")
            if not os.path.exists(path):
                os.makedirs(path)
            data_path = os.path.join(path, f"result_{dataset_type}.pkl")
        else:
            path = os.path.join(PROJECT_ROOT_DIR, args.save_result_dir, "predictions", f"{dataset}", "graphpi_result")
            if not os.path.exists(path):
                os.makedirs(path)
            data_path = os.path.join(path, "result.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(data, f)

    print(f'optimal true proteins: {len_true_proteins}')
    return avg_result_dict, len_true_proteins


def store_results_onemodel(models, test_datasets, device):
    len_true_proteins, result_dict, _ = evaluate_hetero(models, test_datasets, device)

    for dataset, data in result_dict.items():
        if "2016" in dataset:
            dataset_prefix, dataset_type = dataset.split("_")
            data_path = os.path.join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_prefix.lower()}", f"graphpi_result",
                                     f"result_{dataset_type}.pkl")
        else:
            data_path = os.path.join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset}", "graphpi_result", "result.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(data, f)

    print(f'optimal true proteins: {len_true_proteins}')
    return result_dict

@torch.no_grad()
def evaluate(models, datasets, device):
    print(f'Evaluating ...')

    for model in models:
        model.eval()
    result_dict = {}
    len_true_proteins = {}
    for dataset in datasets:
        edge_index = dataset.edge.T.to(device)
        edge_attr = dataset.edge_weights.to(device)
        test_ids = torch.LongTensor(dataset.proteins)

        x = get_node_features(dataset, models, device)
        out = models[0](x, edge_index, edge_attr)
        protein_scores = out.view(-1)[test_ids].cpu().numpy()
        protein_ids = test_ids.cpu().numpy()
        reverse_id_map = {value:key for key,value in dataset.id_map.items()}
        result = {reverse_id_map[protein_id]:score for protein_id, score in zip(protein_ids, protein_scores)}
        result_dict[dataset.dataset] = result

        result = {iprg_converter(protein, dataset): score for protein, score in result.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                            for protein_ in protein_pairs] for protein, protein_pairs in dataset.indistinguishable_groups.items()}

        #result =  adjust_scores_by_indistinguishable_groups(result, indishtinguishable_group)
        true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=indishtinguishable_group)

        len_true_proteins[dataset.dataset] = len(true_proteins)

    return len_true_proteins, result_dict

def get_ce_loss(y_preds, y, loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        loss = F.binary_cross_entropy_with_logits(y_preds, y.type(torch.float))
    elif loss_type == "soft_entropy":
        y_preds = torch.sigmoid(y_preds)
        y_preds = F.relu(y_preds)
        # loss = F.mse_loss(y_preds, y)
        eps = 1e-7
        loss = torch.mean(-(1 - y) * torch.log(1 - y_preds + eps) - y * torch.log(y_preds + eps))
    return loss


def clone(tmp_dict, device):
    result = {}
    for t, tensor in tmp_dict.items():
        result[t] = tensor.clone().detach().to(device)
    return result



def train_hetero(model, data_samplers, device, optimizer, loss_type="cross_entropy"):

    model.train()

    print(f'Training ...')

    train_loss = []
    for dataset, train_ids, _ in tqdm(data_samplers):
        optimizer.zero_grad()

        x_dict = clone(dataset.node_features, device)

        edge_attr_dict = clone(dataset.edge_attr_dict, device)
        edge_dict = clone(dataset.edge_dict, device)
        x_dict = model(x_dict, edge_attr_dict, edge_dict)
        protein_logits = x_dict["protein"].view(-1)

        #x = model(x_dict, dataset.edge_attr.to(device), dataset.edge_index.to(device))
        #protein_output = torch.sigmoid(x.view(-1))
        y = dataset.y.to(device)[train_ids]
        loss = get_ce_loss(protein_logits[train_ids], y, loss_type)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    loss = np.mean(train_loss)
    return loss



@torch.no_grad()
def test_hetero(model, data_samplers, device, loss_type):

    model.eval()

    print(f'Testing ...')

    valid_loss = []
    y_preds = []
    y_true = []
    for dataset, _, test_ids in tqdm(data_samplers):
        y = dataset.y.to(device)[test_ids]
        x_dict = clone(dataset.node_features, device)
        edge_attr_dict = clone(dataset.edge_attr_dict, device)
        edge_dict = clone(dataset.edge_dict, device)
        x_dict = model(x_dict, edge_attr_dict, edge_dict)
        y_logits = x_dict["protein"].view(-1)[test_ids]

        # x = model(x_dict, dataset.edge_attr.to(device), dataset.edge_index.to(device))
        # y_pred = torch.sigmoid(x.view(-1)[test_ids])
        loss = get_ce_loss(y_logits, y, loss_type)
        y_pred = torch.sigmoid(y_logits)

        y_preds.append(y_pred.cpu().numpy())
        y_true.append(y.cpu().numpy())
        valid_loss.append(loss.item())

    valid_loss = np.mean(valid_loss)
    try:
        valid_score = roc_auc_score(np.concatenate(y_true), np.concatenate(y_preds))
    except:
        valid_score = None
    return valid_loss, valid_score


@torch.no_grad()
def evaluate_hetero(model, datasets, device, test_fdr=0.05):
    print(f'Evaluating ...')

    model.eval()

    len_true_proteins = {}
    pauc_dict = {}
    result_dict = {}
    for dataset in datasets:
        x_dict = clone(dataset.node_features, device)
        edge_attr_dict = clone(dataset.edge_attr_dict, device)
        edge_dict = clone(dataset.edge_dict, device)
        x_dict = model(x_dict, edge_attr_dict, edge_dict)
        protein_scores = torch.sigmoid(x_dict["protein"].view(-1))[dataset.proteins].cpu().numpy()

        # x = model(x_dict, dataset.edge_attr.to(device), dataset.edge_index.to(device))
        # protein_scores = torch.sigmoid(x.view(-1))[dataset.proteins].cpu().numpy()
        protein_ids = dataset.proteins #.cpu().numpy()
        reverse_id_map = {value:key for key,value in dataset.id_map.items()}
        result = {reverse_id_map[protein_id]: score for protein_id, score in zip(protein_ids, protein_scores)}
        result_dict[dataset.dataset] = result

        result = {iprg_converter(protein, dataset): score for protein, score in result.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {
            iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                                               for protein_ in protein_pairs] for protein, protein_pairs
            in dataset.indistinguishable_groups.items()}

        true_proteins = get_proteins_by_fdr(result, fdr=test_fdr, contaminate_proteins=contaminate_proteins,
                                            indishtinguishable_group=indishtinguishable_group)

        no_decoy_result = {protein: score for protein, score in result.items() if not protein.startswith('DECOY')}
        y_true = np.array([1 if protein in true_proteins else 0 for protein in no_decoy_result.keys()])
        y_pred = np.array([no_decoy_result[protein] for protein in no_decoy_result.keys()])
        pauc = compute_pauc(y_true, y_pred)
        pauc_dict[dataset.dataset] = pauc
        len_true_proteins[dataset.dataset] = len(true_proteins)
    return len_true_proteins, result_dict, pauc_dict


@torch.no_grad()
def evaluate_hetero_batch(model, data_samplers, device, test_fdr=0.05):
    print(f'Evaluating ...')
    model.eval()

    len_true_proteins = {}
    result_dict = {}

    for dataset, data_loader in tqdm(data_samplers):

        protein_scores = []
        protein_ids = []
        for batch in data_loader:
            batch = batch.to(device)
            batch_size = batch["protein"].batch_size
            out = model(batch.x_dict, batch.edge_attr_dict, batch.edge_index_dict)
            y_pred = torch.sigmoid(out["protein"].view(-1)[:batch_size])
            protein_id = batch["protein"].n_id.cpu().numpy()[:batch_size]
            protein_scores.append(y_pred.cpu().numpy())
            protein_ids.append(protein_id)

        protein_ids = np.concatenate(protein_ids, axis=0)
        protein_scores = np.concatenate(protein_scores, axis=0)

        reverse_id_map = {value:key for key,value in dataset.protein_map.items()}
        result = {reverse_id_map[protein_id]: score for protein_id, score in zip(protein_ids, protein_scores)}
        result_dict[dataset.dataset] = result

        result = {iprg_converter(protein, dataset): score for protein, score in result.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {
            iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                                               for protein_ in protein_pairs] for protein, protein_pairs
            in dataset.indistinguishable_groups.items()}

        true_proteins = get_proteins_by_fdr(result, fdr=test_fdr, contaminate_proteins=contaminate_proteins,
                                            indishtinguishable_group=indishtinguishable_group)

        len_true_proteins[dataset.dataset] = len(true_proteins)
    return len_true_proteins, result_dict


@torch.no_grad()
def test_hetero_batch(model, data_samplers, device, loss_type="cross_entropy"):
    print(f'Validating ...')
    model.eval()

    valid_loss = []
    y_preds = []
    y_true = []
    for dataset, _, data_loader in tqdm(data_samplers):
        for batch in data_loader:
            batch = batch.to(device)
            batch_size = batch["protein"].batch_size
            out = model(batch.x_dict, batch.edge_attr_dict, batch.edge_index_dict)
            y_pred = torch.sigmoid(out["protein"].view(-1)[:batch_size])
            y = batch["protein"].y[:batch_size]
            loss = get_ce_loss(y_pred, y, loss_type)
            valid_loss.append(loss.item())
            y_preds.append(y_pred.cpu().numpy())
            y_true.append(y.cpu().numpy())
            valid_loss.append(loss.item())
    valid_loss = np.mean(valid_loss)
    try:
        valid_score = roc_auc_score(np.concatenate(y_true), np.concatenate(y_preds))
    except:
        valid_score = None
    return valid_loss, valid_score


def train_hetero_batch(model, data_samplers, device, optimizer, loss_type="cross_entropy"):

    print(f'Training ...')
    model.train()
    train_loss = []
    for dataset, data_loader, _ in tqdm(data_samplers):
        for batch in data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch["protein"].batch_size
            out = model(batch.x_dict, batch.edge_attr_dict, batch.edge_index_dict)
            protein_output = torch.sigmoid(out["protein"].view(-1)[:batch_size])
            loss = get_ce_loss(protein_output, batch["protein"].y[:batch_size], loss_type)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
    loss = np.mean(train_loss)
    return loss


if __name__ == "__main__":
    main()