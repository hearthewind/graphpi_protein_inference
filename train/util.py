import torch
import os
import subprocess
import numpy as np
import torch.optim as optim
from torch_geometric.data import HeteroData
from configs import PROJECT_ROOT_DIR
from typing import List

from datasets.hela_3t3 import Hela3T3
from datasets.iPRG2016_twospicies import IPRG2016TS
from datasets.yeast import Yeast
from datasets.iPRG2016 import IPRG2016
from datasets.ups2 import UPS2
from datasets.humanDC import HumanDC
from datasets.pxd import PXD
from datasets.mix18 import Mix18
from models.gnn_model import GNNModel, GNNStack
from torch_geometric.loader import NeighborLoader
import numpy as np
from sklearn.metrics import roc_curve


def compute_pauc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    tpr_interval = tpr[(fpr >= 0.01) & (fpr <= 0.05)]
    fpr_interval = fpr[(fpr >= 0.01) & (fpr <= 0.05)]
    pauc = np.trapz(tpr_interval, fpr_interval)
    return pauc


def iprg_converter(protein, data):
    if isinstance(data, IPRG2016):
        if protein in data.random_proteins:
            return "CONTAMINATE_"+protein
        elif protein in data.a_proteins:
            return "A_"+protein
        elif protein in data.b_proteins:
            return "B_"+protein

    return protein


def get_sampler_hetero(dataset, node_idx, batch_size, fan_out=[-1, -1, -1, -1], rebalance=True):

    #train_proteins, valid_proteins = masks
    #train_mask = dataset.proteins == train_proteins
    #valid_mask = dataset.proteins == valid_proteins

    data = HeteroData()
    for t in ["protein", "peptide", "spectra"]:
        data[t].x = dataset.node_features[t]
        if t == "protein":
            data[t].y = dataset.y
            data[t].batch_size = batch_size
            data[t].n_id = torch.LongTensor(list(dataset.protein_map.values()))

    for t in [("protein", "contain", "peptide"), ("peptide", "contain", "spectra"), ("peptide", "rev_contain", "protein"), ("spectra", "rev_contain", "peptide")]:
        data[t[0], t[1], t[2]].edge_index = dataset.edge_dict[t]
        data[t[0], t[1], t[2]].edge_attr = dataset.edge_attr_dict[t]

    # if rebalance and isinstance(dataset.y, torch.LongTensor):
    #     sampler = ImbalancedSampler(data, input_nodes=node_idx)
    # else:
    #     sampler = None
    train_sampler = NeighborLoader(data, input_nodes=("protein", node_idx), num_neighbors={key: fan_out for key in data.edge_types},
                                    batch_size=batch_size, sampler=None,
                                    num_workers=0)
    return train_sampler


def get_node_features(dataset, models, device):
    model, protein_transform, peptide_transform, spectra_transform, onehot_transform = models
    protein_features, peptide_features, spectra_features = dataset.node_features["protein_features"], \
                                                           dataset.node_features["peptide_features"], \
                                                           dataset.node_features["spectra_features"]

    protein_embed = onehot_transform(protein_features.type(torch.int).to(device)).squeeze(1)
    peptide_embed = onehot_transform(peptide_features.type(torch.int).to(device)).squeeze(1)
    spectra_embed = spectra_transform(spectra_features.to(device))
    x = torch.cat([protein_embed, peptide_embed, spectra_embed])
    return x


def get_model(args, device, data):
    model = GNNModel(args.node_hidden_dim, args.node_hidden_dim, 1, args.num_gnn_layers, gnn_type=args.gnn_type).to(device)

    onehot_transform = torch.nn.Embedding(2, embedding_dim=args.node_hidden_dim).to(device)

    protein_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["protein_features"].shape[1], args.node_hidden_dim),\
                                    torch.nn.ReLU()).to(device)
    peptide_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["peptide_features"].shape[1], args.node_hidden_dim),\
                                    torch.nn.ReLU()).to(device)

    spectra_transform = torch.nn.Sequential(torch.nn.Linear(data.node_features["spectra_features"].shape[1], args.node_hidden_dim),\
                                    torch.nn.ReLU()).to(device)
    models = [model, protein_transform, peptide_transform, spectra_transform, onehot_transform]

    return models


def get_model_hetero(args, device, data):
    metadata = {}
    edge_mode_dict = {}
    for edge_type in data.edge_attr_dict:
        if edge_type[0] in ["peptide", "protein"] and edge_type[2] in ["protein", "peptide"]:
            edge_mode_dict[edge_type] = args.edge_mode#args.edge_mode # Use the pep score directly as the attention score
        else:
            edge_mode_dict[edge_type] = 7#args.edge_mode#args.edge_mode#args.edge_mode#args.edge_mode

    metadata["edge"] = {edge_type:(edge_attr.shape[-1], edge_mode_dict[edge_type]) for edge_type, edge_attr in data.edge_attr_dict.items()}
    metadata["node"] = {node_type:node_feature.shape[-1] for node_type, node_feature in data.node_features.items()}
    model = GNNStack(args.node_hidden_dim, args.edge_hidden_dim, args.edge_mode, args.model_types, args.dropout, args.gnn_activation, \
                     args.concat_states, args.post_hiddens, args.norm_embs, args.aggr, metadata).to(device)
    return model


def get_model_hetero_single(args, device, data):
    metadata = {}

    metadata["node"] = {node_type:node_feature.shape[-1] for node_type, node_feature in data.node_features.items()}

    model = GNNStack(args.node_hidden_dim, args.edge_hidden_dim, args.edge_mode, args.model_types, args.dropout, args.gnn_activation, \
                     args.concat_states, args.post_hiddens, args.norm_embs, args.aggr, metadata).to(device)
    return model


def get_dataset(dataset="HumanDC", train=True, prior=True, prior_offset=0.9, protein_label_type = "benchmark", loss_type = "cross_entropy", pretrain_data_name="epifany", process=True, use_deeppep=False):

    #prior = False if train is True else True
    if dataset == "yeast":
        data = Yeast(protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset.startswith("iPRG2016TS"):
        data_type = dataset.split("_")[1]
        data = IPRG2016TS(data_type=data_type, protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset.startswith("iPRG2016"):
        data_type = dataset.split("_")[1]
        data = IPRG2016(data_type=data_type, protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset.startswith("ups2"):
        data = UPS2(protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset == "humanDC":
        data = HumanDC(protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset == "18mix":
        data = Mix18(protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset.startswith("PXD"):
        data_code = dataset.strip("PXD")
        data = PXD(data_code = data_code, protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)
    elif dataset == 'hela_3t3':
        data = Hela3T3(protein_label_type=protein_label_type, prior=prior, prior_offset=prior_offset, train=train, output_type=loss_type, pretrain_data_name = pretrain_data_name)

    if use_deeppep:
        data.use_deeppep = True
    else:
        data.use_deeppep = False

    if process:
        data.preprocess()
    return data


# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory


def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda


def get_device():
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        cuda = 0 # TODO(m) delete this line
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    # device = torch.device('cpu') # TODO(m) delete this line
    return device

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer


def save_torch_algo(model, out_dir):
    saved_model_filename = os.path.join(PROJECT_ROOT_DIR, "trained_model", out_dir)
    if not os.path.exists(saved_model_filename):
        os.makedirs(saved_model_filename)

    if isinstance(model, List):
        #saved_model_filename = []
        for k in range(len(model)):
            model_filename = os.path.join(saved_model_filename, f"model_{k}.pth")
            #torch.save(model[k], model_filename)
            torch.save(model[k].state_dict(), model_filename)
    else:
        torch.save(model.state_dict(), os.path.join(saved_model_filename, "model.pth"))

# def load_torch_algo(out_dir, models, device, eval=False):
#
#     saved_model_filename = os.path.join(PROJECT_ROOT_DIR, "trained_model", out_dir)
#     paths = sorted(os.listdir(saved_model_filename))
#     if len(paths)>1:
#         for i, _ in enumerate(models):
#             models[i].load_state_dict(torch.load(os.path.join(saved_model_filename, paths[i]), map_location=device))
#         #models = [models[index].load_state_dict(torch.load(os.path.join(saved_model_filename, path))) for index, path in enumerate(paths)]
#         # if eval:
#         #     [model.eval() for model in model]
#     else:
#         models = models.load_state_dict(torch.load(os.path.join(saved_model_filename, paths[0]), map_location=device))  # .to(device)
#         # if eval:
#         #     model.eval()
#     return models

def load_torch_algo(out_dir, models, device="cpu"):

    saved_model_filename = os.path.join(PROJECT_ROOT_DIR, "trained_model", out_dir)
    paths = sorted(os.listdir(saved_model_filename))
    if isinstance(models, list):
        for i, _ in enumerate(models):
            checkpoint = torch.load(os.path.join(saved_model_filename, paths[i]), map_location=device)
            models[i].load_state_dict(checkpoint)  #.to(device)
    else:
        checkpoint = torch.load(os.path.join(saved_model_filename, paths[0]), map_location=device)
        models.load_state_dict(checkpoint)  #.to(device)
    return models

