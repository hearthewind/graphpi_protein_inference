import gc

from train.util import get_device, build_optimizer
from datasets.util import get_proteins_by_fdr
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from models.gnn_model import GNNModel
from configs import DATA_LIST, TEST_DATA
from train.util import get_dataset, iprg_converter


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
    parser.add_argument('--node_hidden_dim', type=int, default=64)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--prior', type=bool, default = True)
    parser.add_argument('--with_spectra', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    return args

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    data_samplers = []
    for dataset in DATA_LIST:
        if dataset in TEST_DATA:
            data = get_dataset(dataset, train=True, prior=False, protein_label_type="decoy")
        else:
            data = get_dataset(dataset, train=True, prior=False, protein_label_type="benchmark")
        num_valid = int(len(data.proteins)*0.2)
        train_node_idx = np.array(data.proteins)
        np.random.shuffle(train_node_idx)
        valid_proteins = torch.LongTensor(train_node_idx[:num_valid])
        train_proteins = torch.LongTensor(train_node_idx[num_valid:])

        valid_sampler = get_sampler(data, valid_proteins, args.batch_size, fan_out=[-1]*args.num_gnn_layers)
        train_sampler = get_sampler(data, train_proteins, args.batch_size, fan_out=[-1]*args.num_gnn_layers)

        data_samplers.append((data, train_sampler, valid_sampler))
        gc.collect()

    # test_data = get_dataset(TEST_DATA, train=False, prior=args.prior)
    # test_sampler = get_sampler(test_data, torch.LongTensor(test_data.proteins), args.batch_size, fan_out=[-1]*args.num_gnn_layers, rebalance=False)
    # reverse_id_map = {value:key for key,value in test_data.id_map.items()}
    # indishtinguishable_group = {reverse_id_map[protein]: [reverse_id_map[protein_] \
    #                                     for protein_ in protein_pairs] for protein, protein_pairs in test_data.indistinguishable_groups.items()}
    #
    # #protein_scores = adjust_scores_by_indistinguishable_groups(test_data.protein_scores, indishtinguishable_group)
    # print(f"benchmark score (EPIPHANY) on dataset ({test_data.dataset}) is:", \
    #       len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=indishtinguishable_group)))
    # test_proteins = torch.LongTensor(test_data.proteins)

    test_samplers = []
    test_datasets = []
    for test_data in TEST_DATA:
        test_dataset = get_dataset(test_data, train=False, prior=args.prior)
        test_datasets.append(test_dataset)
        test_samplers.append(get_sampler(test_dataset, torch.LongTensor(test_dataset.proteins), args.batch_size,
                                   fan_out=[-1] * args.num_gnn_layers, rebalance=False))

    #protein_scores = adjust_scores_by_indistinguishable_groups(test_data.protein_scores, indishtinguishable_group)
    for test_data in test_datasets:
        reverse_id_map = {value: key for key, value in test_data.id_map.items()}
        indishtinguishable_group = {reverse_id_map[protein]: [reverse_id_map[protein_] \
                                                              for protein_ in protein_pairs] for protein, protein_pairs
                                    in test_data.indistinguishable_groups.items()}

        print(f"benchmark score (EPIPHANY) on dataset ({test_data.dataset}) is:", \
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.05, indishtinguishable_group=indishtinguishable_group)))


    models = get_model(args, device, test_data, with_spectra=args.with_spectra)

    # build optimizer
    parameters = []
    for model in models:
        parameters.extend(list(model.parameters()))
    scheduler, optimizer = build_optimizer(args, parameters)


    best_valid_score = np.inf
    for epoch in range(args.epochs):
        train_loss = train_batch(models, data_samplers, device, optimizer, epoch)
        valid_loss = test_batch(models, data_samplers, device)

        len_true_proteins = evaluate_batch(models, test_datasets, test_samplers, device)

        if best_valid_score > valid_loss:
            best_true_proteins = len_true_proteins
            #best_protein_list = result
            best_valid_score = valid_loss
            #save_torch_algo(models, TEST_DATA)

        if epoch % 1 == 0:

            print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
                  f'Num true proteins: {len_true_proteins}, optimal true proteins: {best_true_proteins}')
        # else:
        #     print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
        #           f'Val score: {valid_score:.4f}.')



def get_model(args, device, data, with_spectra=True):
    if with_spectra:
        model = GNNModel(args.node_hidden_dim, args.node_hidden_dim, 1, args.num_gnn_layers, gnn_type=args.gnn_type).to(device)

        onehot_transform = torch.nn.Embedding(2, embedding_dim=args.node_hidden_dim).to(device)

        protein_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["protein_features"].shape[1], args.node_hidden_dim),\
                                        torch.nn.ReLU()).to(device)
        peptide_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["peptide_features"].shape[1], args.node_hidden_dim),\
                                        torch.nn.ReLU()).to(device)

        spectra_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["spectra_features"].shape[1], args.node_hidden_dim),\
                                        torch.nn.ReLU()).to(device)
        models = [model, protein_transform, peptide_transform, spectra_transform, onehot_transform]
    else:
        model = GNNModel(args.node_hidden_dim, args.node_hidden_dim, 1, args.num_gnn_layers, gnn_type=args.gnn_type).to(device)

        protein_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["protein_features"].shape[1], args.node_hidden_dim),\
                                        torch.nn.ReLU()).to(device)
        peptide_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["peptide_features"].shape[1], args.node_hidden_dim),\
                                        torch.nn.ReLU()).to(device)
        models = [model, protein_transform, peptide_transform]

    return models


def get_sampler(dataset, node_idx, batch_size, fan_out=[-1, -1, -1, -1], rebalance=True):

    #x = torch.tensor(dataset.node_features)
    edge_index = dataset.edge.T
    edge_weights = dataset.edge_weights
    n_id = torch.LongTensor(list(dataset.id_map.values()))

    protein_features = dataset.node_features["protein_features"]
    peptide_features = dataset.node_features["peptide_features"]
    spectra_features = dataset.node_features["spectra_features"]

    len_protein, len_peptide, len_spectra = len(protein_features), len(peptide_features), len(spectra_features)
    protein_features = torch.nn.functional.pad(protein_features, (0, 0, 0, len(n_id) - len_protein))
    peptide_features = torch.nn.functional.pad(peptide_features, (0, 0, len_protein, len(n_id) - len_protein - len_peptide))
    spectra_features = torch.nn.functional.pad(spectra_features, (0, 0, len_protein+len_peptide, 0))

    node_types = torch.cat([torch.zeros(size=(len_protein, )), torch.ones(size=(len_peptide, )), 2*torch.ones(size=(len_spectra, ))])
    data = Data(y=dataset.y, edge_index=edge_index, n_id=n_id, node_types = node_types, edge_attr=edge_weights, protein_features = protein_features, peptide_features = peptide_features, spectra_features = spectra_features)

    if rebalance and isinstance(dataset.y, torch.LongTensor):
        sampler = ImbalancedSampler(data, input_nodes=node_idx)
    else:
        sampler = None
    train_sampler = NeighborLoader(data, input_nodes=node_idx, num_neighbors=fan_out,
                                    batch_size=batch_size, sampler=sampler,
                                    num_workers=0)
    return train_sampler



@torch.no_grad()
def evaluate_batch(models, datasets, test_samplers, device):
    for model in models:
        model.eval()

    len_true_proteins = {}
    for dataset, test_sampler in zip(datasets, test_samplers):
        pbar = tqdm(total=len(test_sampler))
        pbar.set_description(f'Evaluating ...')

        protein_scores = []
        protein_ids = []
        for batch_data in test_sampler:
            x = get_node_features_batch(batch_data, models, device)
            out = models[0](x, batch_data.edge_index.to(device), batch_data.edge_attr.to(device))
            protein_score = out.view(-1).cpu().numpy()[:batch_data.batch_size]
            protein_id = batch_data.n_id.cpu().numpy()[:batch_data.batch_size]
            protein_scores.append(protein_score)
            protein_ids.append(protein_id)
            pbar.update(1)
        torch.cuda.empty_cache()
        pbar.close()

        protein_ids = np.concatenate(protein_ids, axis=0)
        protein_scores = np.concatenate(protein_scores, axis=0)

        reverse_id_map = {value:key for key,value in dataset.id_map.items()}
        result = {reverse_id_map[protein_id]:score for protein_id, score in zip(protein_ids, protein_scores)}

        result = {iprg_converter(protein, dataset): score for protein, score in result.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                            for protein_ in protein_pairs] for protein, protein_pairs in dataset.indistinguishable_groups.items()}

        true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=indishtinguishable_group)
        psuedo_true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=None)
        len_true_proteins[dataset.dataset] = len(true_proteins)

    return len_true_proteins


def get_node_features_batch(batch_data, models, device):
    model, protein_transform, peptide_transform, spectra_transform, onehot_transform = models

    hidden_size = spectra_transform[0].weight.shape[0]
    protein_features = batch_data.protein_features
    peptide_features = batch_data.peptide_features
    spectra_features = batch_data.spectra_features
    node_types = batch_data.node_types
    node_embeds = torch.zeros(size=(len(protein_features), hidden_size)).to(device)
    node_embeds[node_types == 0,:] = protein_transform(protein_features[node_types == 0, :].to(device))
    node_embeds[node_types == 1,:] = peptide_transform(peptide_features[node_types == 1, :].to(device))
    node_embeds[node_types == 2,:] = spectra_transform(spectra_features[node_types == 2, :].to(device))

    return node_embeds


def adjust_scores_by_indistinguishable_groups(score_dict, indishtinguishable_group):
    for protein, score in score_dict.items():
        if protein in indishtinguishable_group:
            other_indis_proteins = indishtinguishable_group[protein]
            score_dict[protein] = max(score, *[score_dict[indis_protein] for indis_protein in other_indis_proteins])
    return score_dict

@torch.no_grad()
def test_batch(models, data_samplers, device):

    for model in models:
        model.eval()

    print(f'Testing ...')

    valid_loss = []
    y_preds = []
    y_trues = []
    for dataset, _, test_sampler in tqdm(data_samplers):

        for batch_data in test_sampler:
            x = get_node_features_batch(batch_data, models, device)
            out = models[0](x, batch_data.edge_index.to(device), batch_data.edge_attr.to(device))
            y_pred = out.view(-1)[:batch_data.batch_size]
            y = batch_data.y[:batch_data.batch_size].to(device)
            loss = get_rank_loss(y_pred, y)
            protein_score = out.view(-1).cpu().numpy()[:batch_data.batch_size]
            y_preds.append(protein_score)
            y_trues.append(y.cpu().numpy())
            valid_loss.append(loss.item())

    valid_loss = np.mean(valid_loss)
    #valid_score = roc_auc_score(np.concatenate(y_trues), np.concatenate(y_preds))

    return valid_loss#, valid_score


def train_batch(models, data_samplers, device, optimizer, epoch):

    for model in models:
        model.train()

    print(f'Epoch {epoch:02d}')

    train_loss = []
    for dataset, train_sampler, _ in tqdm(data_samplers):
        for batch_data in train_sampler:
            x = get_node_features_batch(batch_data, models, device)
            out = models[0](x, batch_data.edge_index.to(device), batch_data.edge_attr.to(device))
            optimizer.zero_grad()
            y = batch_data.y[:batch_data.batch_size].to(device)
            loss = get_rank_loss(out.view(-1)[:batch_data.batch_size], y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

    loss = np.mean(train_loss)
    return loss

loss_fn = torch.nn.MarginRankingLoss(margin=1)
def get_rank_loss(y_preds, y, loss_type="soft_entropy"):
    if loss_type == "cross_entropy":
        loss = F.binary_cross_entropy(y_preds, y.type(torch.float))
    elif loss_type == "soft_entropy":
        y_preds = F.relu(y_preds)
        eps = 1e-7
        loss = torch.mean(-(1 - y) * torch.log(1 - y_preds + eps) - y * torch.log(y_preds + eps))
    return loss



if __name__ == "__main__":
    main()