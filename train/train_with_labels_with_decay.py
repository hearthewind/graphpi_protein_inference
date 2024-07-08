import gc
import torch.optim as optim
from train.util import get_device, build_optimizer, load_torch_algo
from datasets.util import get_proteins_by_fdr
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from configs import DATA_LIST, TEST_DATA
from sklearn.metrics import roc_auc_score
from train.util import save_torch_algo, get_dataset, get_model, get_node_features, iprg_converter

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
    parser.add_argument('--prior_offset', type=float, default = 0.9)
    parser.add_argument('--loss_type', type=str, default="cross_entropy")
    parser.add_argument('--pretrain_data_name', type=str, default="epifany")

    args = parser.parse_args()
    print(args)
    return args

def main():

    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_samplers = []
    for dataset in DATA_LIST:
        if dataset in TEST_DATA:
            data = get_dataset(dataset, train=True, prior=args.prior, prior_offset=args.prior_offset, protein_label_type="decoy", pretrain_data_name=args.pretrain_data_name, loss_type=args.loss_type)
        else:
            data = get_dataset(dataset, train=True, prior=args.prior, prior_offset=args.prior_offset, protein_label_type="decoy_sampling", pretrain_data_name=args.pretrain_data_name, loss_type=args.loss_type)
        num_valid = int(len(data.proteins)*0.2)
        train_node_idx = np.array(data.proteins)
        np.random.shuffle(train_node_idx)
        valid_proteins = torch.LongTensor(train_node_idx[:num_valid])
        train_proteins = torch.LongTensor(train_node_idx[num_valid:])
        data_samplers.append((data, train_proteins, valid_proteins))
        gc.collect()

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
              len(get_proteins_by_fdr(test_data.protein_scores, contaminate_proteins=test_data.get_contaminate_proteins(), fdr=0.01, indishtinguishable_group=indishtinguishable_group)))
        #test_proteins = torch.LongTensor(test_data.proteins)

    device = get_device()

    models = get_model(args, device, test_data)
    # models = load_torch_algo("pretrain_small", models)
    # models = [model.to(device) for model in models]
    # models = [model.eval() for model in models]
    print(evaluate(models, test_datasets, device))
    #print(evaluate(models, test_datasets, device))

    # build optimizer
    parameters = []
    for model in models:
        parameters.extend(list(model.parameters()))
    scheduler, optimizer = build_optimizer(args, parameters)

    model_dir = f"decay_small_noniso_{args.loss_type}_{args.pretrain_data_name}_prior-{args.prior}_offset-{args.prior_offset}"

    best_valid_score = 0
    for epoch in range(args.epochs):
        train_loss = train(models, data_samplers, device, optimizer, epoch, args.loss_type)
        valid_loss, valid_score = test(models, data_samplers, device, args.loss_type)

        len_true_proteins = evaluate(models, test_datasets, device)

        if best_valid_score < valid_score:
            best_true_proteins = len_true_proteins
            #best_protein_list = result
            best_valid_score = valid_score
            save_torch_algo(models, model_dir)
            #save_torch_algo(models, f"original_pretrain_small_noniso_{args.loss_type}_{args.pretrain_data_name}_prior-{args.prior}_offset-{args.prior_offset}")



        #if epoch % 10 == 0:

        print(f'Epoch: {epoch:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
              f'Val score: {valid_score:.4f}. Num true proteins: {len_true_proteins}, optimal true proteins: {best_true_proteins}')

        # if epoch % 50 == 0:
        #     models = load_torch_algo("pretrain_small", models)
        #     models = [model.to(device) for model in models]
        #     models = [model.eval() for model in models]
        #     print(evaluate(models, test_datasets, device))

    filter_fn = filter(lambda p: p.requires_grad, parameters)
    optimizer = optim.Adam(filter_fn, lr=args.lr/10.0, weight_decay=args.weight_decay)

    models = load_torch_algo(model_dir, models, device)
    models = [model.to(device) for model in models]
    print("Decaying Learning Rate")

    for epoch in range(args.epochs):
        train_loss = train(models, data_samplers, device, optimizer, epoch, args.loss_type)
        valid_loss, valid_score = test(models, data_samplers, device, args.loss_type)

        len_true_proteins = evaluate(models, test_datasets, device)

        if best_valid_score < valid_score:
            best_true_proteins = len_true_proteins
            # best_protein_list = result
            best_valid_score = valid_score
            save_torch_algo(models, model_dir)
            # save_torch_algo(models, f"original_pretrain_small_noniso_{args.loss_type}_{args.pretrain_data_name}_prior-{args.prior}_offset-{args.prior_offset}")

        # if epoch % 10 == 0:

        print(f'Epoch: {epoch + args.epochs:04d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, '
              f'Val score: {valid_score:.4f}. Num true proteins: {len_true_proteins}, optimal true proteins: {best_true_proteins}')

@torch.no_grad()
def evaluate(models, datasets, device):
    print(f'Evaluating ...')

    for model in models:
        model.eval()

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

        result = {iprg_converter(protein, dataset): score for protein, score in result.items()}
        contaminate_proteins = [iprg_converter(protein, dataset) for protein in dataset.get_contaminate_proteins()]
        indishtinguishable_group = {iprg_converter(reverse_id_map[protein], dataset): [iprg_converter(reverse_id_map[protein_], dataset) \
                                            for protein_ in protein_pairs] for protein, protein_pairs in dataset.indistinguishable_groups.items()}

        #result =  adjust_scores_by_indistinguishable_groups(result, indishtinguishable_group)
        true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=indishtinguishable_group)
        psuedo_true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=contaminate_proteins, indishtinguishable_group=None)

        len_true_proteins[dataset.dataset] = len(true_proteins)
    #psuedo_true_proteins = get_proteins_by_fdr(result, fdr=0.05, contaminate_proteins=None)
    return len_true_proteins#, result


def adjust_scores_by_indistinguishable_groups(score_dict, indishtinguishable_group):
    for protein, score in score_dict.items():
        if protein in indishtinguishable_group:
            other_indis_proteins = indishtinguishable_group[protein]
            score_dict[protein] = max(score, *[score_dict[indis_protein] for indis_protein in other_indis_proteins])
    return score_dict

@torch.no_grad()
def test(models, data_samplers, device, loss_type="cross_entropy"): #TODO(m) information leak between train and valid

    for model in models:
        model.eval()

    print(f'Testing ...')

    valid_loss = []
    y_preds = []
    y_true = []

    for dataset, _, test_ids in tqdm(data_samplers):
        y = dataset.y.to(device)[test_ids]
        x = get_node_features(dataset, models, device)
        out = models[0](x, dataset.edge.T.to(device), dataset.edge_weights.to(device))

        y_pred = out.view(-1)[test_ids]

        loss = get_ce_loss(y_pred, y, loss_type)
        #single_out = models[0](x, dataset.single_edge.T.to(device), dataset.single_edge_weights.to(device))
        #kept_single_proteins = list(set(test_ids.cpu().numpy()).intersection(set(dataset.single_proteins)))
        #single_contrast_loss = get_ranking_loss(single_out.view(-1)[kept_single_proteins], out.view(-1)[kept_single_proteins], margin=0.1)

        # one_hit_out = models[0](x, dataset.one_hit_edge.T.to(device), dataset.one_hit_edge_weights.to(device))
        # kept_one_hit_proteins = list(set(test_ids.cpu().numpy()).intersection(set(dataset.one_hit_proteins)))
        # one_hit_contrast_loss = get_ranking_loss(out.view(-1)[kept_one_hit_proteins], one_hit_out.view(-1)[kept_one_hit_proteins], margin=0.1)
        # loss = loss + one_hit_contrast_loss
        #loss = loss + single_contrast_loss
        # #loss = loss + single_contrast_loss + one_hit_contrast_loss
        # kept_single_proteins = list(set(test_ids.cpu().numpy()).intersection(set(dataset.single_proteins)))
        # contrast_loss = get_ranking_loss(single_out.view(-1)[kept_single_proteins], out.view(-1)[kept_single_proteins])
        # loss = loss + single_contrast_loss
        y_preds.append(y_pred.cpu().numpy())
        y_true.append(y.cpu().numpy())
        valid_loss.append(loss.item())
        #torch.cuda.empty_cache()

    valid_loss = np.mean(valid_loss)
    valid_score = roc_auc_score(np.concatenate(y_true), np.concatenate(y_preds))

    return valid_loss, valid_score


def train(models, data_samplers, device, optimizer, epoch, loss_type="cross_entropy"):

    for model in models:
        model.train()

    print(f'Epoch {epoch:02d}')

    train_loss = []
    for dataset, train_ids, _ in tqdm(data_samplers):
        y = dataset.y.to(device)[train_ids]
        optimizer.zero_grad()
        x = get_node_features(dataset, models, device)
        out = models[0](x, dataset.edge.T.to(device), dataset.edge_weights.to(device))

        loss = get_ce_loss(out.view(-1)[train_ids], y, loss_type)
        # if epoch%20 == 0:
        #    dataset.generate_edge_mapping_without_shared_proteins()
        # single_out = models[0](x, dataset.single_edge.T.to(device), dataset.single_edge_weights.to(device))
        # kept_single_proteins = list(set(train_ids.cpu().numpy()).intersection(set(dataset.single_proteins)))
        # single_contrast_loss = get_ranking_loss(single_out.view(-1)[kept_single_proteins], out.view(-1)[kept_single_proteins], margin=0.1)

        # if epoch%20 == 0:
        #    #dataset.generate_edge_mapping_without_shared_proteins()
        #    dataset.generate_edge_mapping_with_one_hit_proteins()
        # 
        # one_hit_out = models[0](x, dataset.one_hit_edge.T.to(device), dataset.one_hit_edge_weights.to(device))
        # kept_one_hit_proteins = list(set(train_ids.cpu().numpy()).intersection(set(dataset.one_hit_proteins)))
        # one_hit_contrast_loss = get_ranking_loss(out.view(-1)[kept_one_hit_proteins],one_hit_out.view(-1)[kept_one_hit_proteins], margin=0.1)
        # loss = loss + one_hit_contrast_loss
        #loss = loss + single_contrast_loss
        # #loss = loss + 2*single_contrast_loss + one_hit_contrast_loss

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    loss = np.mean(train_loss)
    return loss

#loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
def get_ranking_loss(pos_preds, neg_preds, margin=0.1):
    #cf_loss = (-1.0) * F.logsigmoid(pos_preds - neg_preds)
    #loss = torch.mean(cf_loss)
    # loss_fn = torch.nn.MarginRankingLoss(margin=1)
    loss = torch.nn.MarginRankingLoss(margin=margin)(pos_preds, neg_preds, target=torch.ones(len(pos_preds)).to(pos_preds.device))

    return loss

def get_ce_loss(y_preds, y, loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        loss = F.binary_cross_entropy(y_preds, y.type(torch.float))
    elif loss_type == "soft_entropy":
        y_preds = F.relu(y_preds)
        # loss = F.mse_loss(y_preds, y)
        eps = 1e-7
        loss = torch.mean(-(1 - y) * torch.log(1 - y_preds + eps) - y * torch.log(y_preds + eps))
    return loss

# def get_ce_loss(y_preds, y):
#     loss = F.binary_cross_entropy(y_preds, y.type(torch.float))
#     return loss



if __name__ == "__main__":
    main()