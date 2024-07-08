import gc
from train.util import get_device, build_optimizer
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from configs import DATA_LIST, TEST_DATA
from sklearn.metrics import roc_auc_score
from train.util import save_torch_algo, load_torch_algo, get_sampler_hetero as get_sampler
from datasets.util import get_proteins_by_fdr
from train.util import iprg_converter, get_node_features, get_dataset
#from train.util import get_model
from train.util import get_model_hetero as get_model
import random
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
import os
import pickle
from experiment import get_fdr_vs_TP_graphs
import copy
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=500)
    parser.add_argument('--opt_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--node_hidden_dim', type=int, default=100)
    parser.add_argument('--num_gnn_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--prior', type=bool, default = True)
    parser.add_argument('--prior_offset', type=float, default = 0.9)
    parser.add_argument('--loss_type', type=str, default="cross_entropy")
    parser.add_argument('--pretrain_data_name', type=str, default="epifany")
    parser.add_argument('--protein_label_type', type=str, default="benchmark")


    # For hetero GNN
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE_EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default="0",) # default to be 0 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='add',)
    parser.add_argument('--edge_hidden_dim', type=int, default=100)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_deeppep', type=bool, default=False)
    parser.add_argument('--use_pretrain', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    return args

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = get_device()

    save_path = f"pretrain_noniso_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}"

    if args.use_pretrain:
        data = get_dataset("ups2", train=True, prior=args.prior, prior_offset=args.prior_offset,
                           protein_label_type="decoy", pretrain_data_name=args.pretrain_data_name,
                           loss_type=args.loss_type, use_deeppep=args.use_deeppep)
        models = get_model(args, device, data)
        models = load_torch_algo(out_dir=save_path, models=models, device=device)
        fine_tune_decoy_by_dataset(args, models, "iPRG2016_A", device)
        fine_tune_decoy_by_dataset(args, models, "iPRG2016_B", device)
        fine_tune_decoy_by_dataset(args, models, "ups2", device)
        return
    pretrain_multihead_output_modules = {}
    for dataset in DATA_LIST:
        data_samplers = []
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
        else:
            train_sampler = get_sampler(data, train_proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers, rebalance=False)
            valid_sampler = get_sampler(data, valid_proteins, args.batch_size, fan_out=[-1] * args.num_gnn_layers, rebalance=False)
            data_samplers.append((data, train_sampler, valid_sampler))

        pretrain_multihead_output_modules[dataset] = torch.nn.Sequential(torch.nn.Linear(args.node_hidden_dim, args.node_hidden_dim // 2), torch.nn.ReLU(),\
                                                               torch.nn.Linear(args.node_hidden_dim // 2, 1))

    gc.collect()

    models = get_model(args, device, data)
    for _, module in pretrain_multihead_output_modules.items():
        module.to(device)
    pretrain_multihead_parameters = []
    for _, module in pretrain_multihead_output_modules.items():
        pretrain_multihead_parameters.extend(list(module.parameters()))
    scheduler, optimizer = build_optimizer(args, list(models.parameters())+pretrain_multihead_parameters)

    best_valid_score = 0
    best_valid_loss = np.inf

    for epoch in range(args.epochs):

        if args.batch_size <= 0:
            train_loss = train_hetero(models, pretrain_multihead_output_modules, data_samplers, device, optimizer, args.loss_type)
            valid_loss, valid_score = test_hetero(models, pretrain_multihead_output_modules, data_samplers, device, args.loss_type)
        else:
            raise NotImplementedError

        if args.loss_type == "soft_entropy":
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                save_torch_algo(models, save_path)
                best_params = copy.deepcopy(models.state_dict())

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}.')

        elif args.loss_type == "cross_entropy":
            if best_valid_score < valid_score:
                best_valid_score = valid_score
                save_torch_algo(models, save_path)
                best_params = copy.deepcopy(models.state_dict())
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
                  f'Val score: {valid_score:.4f}.')

    models.load_state_dict(best_params)

    fine_tune_decoy_by_dataset(args, models, "iPRG2016_A", device)
    fine_tune_decoy_by_dataset(args, models, "iPRG2016_B", device)
    fine_tune_decoy_by_dataset(args, models, "ups2", device)

    #len_true_proteins, result_dict = evaluate_hetero(models, test_datasets, device)
    # store_results(models, test_datasets, device)
    # get_fdr_vs_TP_graphs()


def fine_tune_decoy_by_dataset(args, models, dataset, device):

    data = get_dataset(dataset, train=True, prior=args.prior, prior_offset=args.prior_offset, protein_label_type="decoy", pretrain_data_name=args.pretrain_data_name, loss_type=args.loss_type, use_deeppep=args.use_deeppep)
    train_node_idx = np.array(list(data.protein_map.values()))
    np.random.shuffle(train_node_idx)
    seed_batchs = np.array_split(train_node_idx, 5)

    reverse_id_map = {value: key for key, value in data.id_map.items()}
    indishtinguishable_group = {reverse_id_map[protein]: [reverse_id_map[protein_] \
                                                          for protein_ in protein_pairs] for protein, protein_pairs
                                in data.indistinguishable_groups.items()}

    print(f"benchmark score ({args.pretrain_data_name}) on dataset ({data.dataset}) is:", \
          len(get_proteins_by_fdr(data.protein_scores, contaminate_proteins=data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=indishtinguishable_group)))


    results = {}
    avg_results = []

    for model_num, seed_batch in enumerate(seed_batchs):
        test_index = seed_batch
        train_index = list(set(train_node_idx) - set(test_index))
        #num_train = int(len(train_index) * 0.8)
        #valid_index = train_index[num_train:]
        #train_index = train_index[:num_train]

        multihead_outputs_module = torch.nn.Sequential(torch.nn.Linear(args.node_hidden_dim, args.node_hidden_dim // 2), torch.nn.ReLU(),\
                                                               torch.nn.Linear(args.node_hidden_dim // 2, 1))

        models.to(device)
        for p in models.parameters(): p.requires_grad = False

        multihead_outputs_module.to(device)
        scheduler, optimizer = build_optimizer(args, multihead_outputs_module.parameters())
        #scheduler, optimizer = build_optimizer(args, list(multihead_outputs_module.parameters())+list(models.parameters()))

        best_valid_score = 0
        for epoch in range(50):
            #train_loss = train_hetero(models, {dataset:multihead_outputs_module}, [(data, train_index, valid_index)], device, optimizer, args.loss_type)
            #valid_loss, valid_score = test_hetero(models, {dataset:multihead_outputs_module}, [(data, train_index, valid_index)], device, args.loss_type)
            train_loss = train_hetero(models, {dataset:multihead_outputs_module}, [(data, train_index, train_index)], device, optimizer, args.loss_type)

            len_true_proteins, probs = evaluate_hetero(models, {dataset:multihead_outputs_module}, [data], device)
            best_probs = probs
            print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, True proteins: {len_true_proteins}.')
            # if args.loss_type == "soft_entropy":
            #     if best_valid_loss > valid_loss:
            #         best_probs = probs
            #         best_valid_loss = valid_loss
            #         #save_torch_algo(models,
            #         #                f"pretrain_noniso_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}")
            #         #best_params = copy.deepcopy(models.state_dict())
            #     print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, True proteins: {len_true_proteins}.')
            #
            # elif args.loss_type == "cross_entropy":
            #     if best_valid_score < valid_score:
            #         best_probs = probs
            #         best_valid_score = valid_score
            #         #save_torch_algo(models,
            #         #                f"pretrain_noniso_{args.loss_type}_{args.pretrain_data_name}_{args.protein_label_type}_prior-{args.prior}_offset-{args.prior_offset}")
            #         #best_params = copy.deepcopy(models.state_dict())
            #     print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
            #           f'Val score: {valid_score:.4f}, True proteins: {len_true_proteins}.')

        results.update({protein_index: prob for protein_index, prob in zip(test_index, best_probs[test_index])})
        avg_results.append(best_probs)

    #avg_results = np.mean(np.array(avg_results), axis=0)
    #avg_results = {protein_index:prob for protein_index, prob in zip(train_node_idx, avg_results[train_node_idx])}
    result = {reverse_id_map[protein_id]:score for protein_id, score in results.items()}
    result = {iprg_converter(protein, data): score for protein, score in result.items()}
    contaminate_proteins = [iprg_converter(protein, data) for protein in data.get_contaminate_proteins()]
    indishtinguishable_group = {
        iprg_converter(reverse_id_map[protein], data): [iprg_converter(reverse_id_map[protein_], data) \
                                                             for protein_ in protein_pairs] for protein, protein_pairs in
        data.indistinguishable_groups.items()}


    true_proteins = get_proteins_by_fdr(result, contaminate_proteins=contaminate_proteins, fdr=0.05,
                                        indishtinguishable_group=indishtinguishable_group)
    decoy_true_proteins = get_proteins_by_fdr(result, contaminate_proteins=[], fdr=0.05,
                                        indishtinguishable_group=indishtinguishable_group)
    print(f"True proteins: {len(true_proteins)}, decoy true proteins: {len(decoy_true_proteins)}")
    return true_proteins



def train_hetero(model, multihead_output_modules, data_samplers, device, optimizer, loss_type="cross_entropy"):

    model.train()
    for dataset, module in multihead_output_modules.items():
        multihead_output_modules[dataset].train()
    #print(f'Training ...')

    train_loss = []
    for dataset, train_ids, _ in data_samplers:

        x_dict = clone(dataset.node_features, device)

        edge_attr_dict = clone(dataset.edge_attr_dict, device)
        edge_dict = clone(dataset.edge_dict, device)
        x_dict = model(x_dict, edge_attr_dict, edge_dict)
        x_dict["protein"] = multihead_output_modules[dataset.dataset](x_dict["protein"])
        protein_output = torch.sigmoid(x_dict["protein"].view(-1))

        y = dataset.y.to(device)[train_ids]
        optimizer.zero_grad()
        loss = get_ce_loss(protein_output[train_ids], y, loss_type)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    loss = np.mean(train_loss)
    return loss


@torch.no_grad()
def test_hetero(model, multihead_output_modules, data_samplers, device, loss_type):

    model.eval()
    for dataset, module in multihead_output_modules.items():
        multihead_output_modules[dataset].eval()

    #print(f'Testing ...')

    valid_loss = []
    y_preds = []
    y_true = []
    for dataset, _, test_ids in data_samplers:
        y = dataset.y.to(device)[test_ids]
        x_dict = clone(dataset.node_features, device)
        edge_attr_dict = clone(dataset.edge_attr_dict, device)
        edge_dict = clone(dataset.edge_dict, device)
        x_dict = model(x_dict, edge_attr_dict, edge_dict)
        x_dict["protein"] = multihead_output_modules[dataset.dataset](x_dict["protein"])
        y_pred = torch.sigmoid(x_dict["protein"].view(-1)[test_ids])

        # x = model(x_dict, dataset.edge_attr.to(device), dataset.edge_index.to(device))
        # y_pred = torch.sigmoid(x.view(-1)[test_ids])
        loss = get_ce_loss(y_pred, y, loss_type)
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
def evaluate_hetero(model, multihead_output_modules, datasets, device, test_fdr=0.05):
    print(f'Evaluating ...')

    model.eval()
    for dataset, module in multihead_output_modules.items():
        multihead_output_modules[dataset].eval()
    len_true_proteins = {}

    result_dict = {}
    for dataset in datasets:
        x_dict = clone(dataset.node_features, device)
        edge_attr_dict = clone(dataset.edge_attr_dict, device)
        edge_dict = clone(dataset.edge_dict, device)
        x_dict = model(x_dict, edge_attr_dict, edge_dict)
        x_dict["protein"] = multihead_output_modules[dataset.dataset](x_dict["protein"])
        protein_scores = torch.sigmoid(x_dict["protein"].view(-1))[list(dataset.protein_map.values())].cpu().numpy()

        # x = model(x_dict, dataset.edge_attr.to(device), dataset.edge_index.to(device))
        # protein_scores = torch.sigmoid(x.view(-1))[dataset.proteins].cpu().numpy()
        protein_ids = dataset.protein_map.values()#.cpu().numpy()
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

        len_true_proteins[dataset.dataset] = len(true_proteins)
    return len_true_proteins, protein_scores




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


def store_results(models, test_datasets, device):
    len_true_proteins, result_dict = evaluate_hetero(models, test_datasets, device)

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


def adjust_scores_by_indistinguishable_groups(score_dict, indishtinguishable_group):
    for protein, score in score_dict.items():
        if protein in indishtinguishable_group:
            other_indis_proteins = indishtinguishable_group[protein]
            score_dict[protein] = max(score, *[score_dict[indis_protein] for indis_protein in other_indis_proteins])
    return score_dict


def get_ce_loss(y_preds, y, loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        loss = F.binary_cross_entropy(y_preds, y.type(torch.float))
    elif loss_type == "soft_entropy":
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

if __name__ == "__main__":
    main()