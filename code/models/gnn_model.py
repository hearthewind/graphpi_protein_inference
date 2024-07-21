import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HeteroConv
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing

from models.encoder import EncoderRNN, EncoderTransformer, EncoderCNN
from models.util import get_activation
from torch.nn.init import xavier_uniform_, zeros_
from collections import defaultdict
from torch_geometric.utils import degree, add_self_loops, add_remaining_self_loops
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class GNNModel(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, gnn_type):
        super().__init__()

        if gnn_type == "GraphSAGE":
            self.gnn_model = SAGEConv
        elif gnn_type == "GCN":
            self.gnn_model = GCNConv

        self.gnn_type = gnn_type
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.gnn_model(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(self.gnn_model(hidden_channels, hidden_channels))
        self.convs.append(self.gnn_model(hidden_channels, out_channels))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            if self.gnn_type == "GraphSAGE":
                x = self.convs[i](x, edge_index)
            elif self.gnn_type == "GCN":
                x = self.convs[i](x, edge_index, edge_weight)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return torch.sigmoid(x)


class GNNStack(torch.nn.Module):

    def __init__(self,
                node_hidden_dim, edge_hidden_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs, aggr, metadata
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.metadata = metadata
        self.concat_states = concat_states
        self.model_types = model_types.split('_')
        self.gnn_layer_num = len(self.model_types)

        if normalize_embs is None:
            normalize_embs = [True, ] * len(self.model_types)
        else:
            normalize_embs = list(map(bool, map(int, normalize_embs.split('_'))))
        if node_post_mlp_hiddens is None:
            node_post_mlp_hiddens = [node_hidden_dim]
        else:
            node_post_mlp_hiddens = list(map(int, node_post_mlp_hiddens.split('_')))
        print(self.model_types, normalize_embs, node_post_mlp_hiddens)

        edge_metadata = metadata["edge"]
        node_metadata = metadata["node"]

        self.onehot_transform = torch.nn.Embedding(2, embedding_dim=node_hidden_dim)#.to(device)

        # self.protein_transform = torch.nn.Sequential(torch.nn.Linear(node_metadata["peptide"], node_hidden_dim), torch.nn.ReLU())
        # self.peptide_transform = torch.nn.Sequential(torch.nn.Linear(node_metadata["peptide"], node_hidden_dim), torch.nn.ReLU())
        self.spectra_transform = torch.nn.Sequential(torch.nn.Linear(node_metadata["spectra"], node_hidden_dim), torch.nn.ReLU())

        # convs
        self.convs = self.build_convs(node_hidden_dim, edge_hidden_dim, edge_mode,
                                    self.model_types, normalize_embs, activation, aggr, metadata)

        # post node update
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_hidden_dim * len(self.model_types)),
                                                          1, node_post_mlp_hiddens,
                                                          dropout, activation, node_metadata)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_hidden_dim, 1, node_post_mlp_hiddens, dropout,
                                                          activation, node_metadata)

        self.edge_update_mlps = self.build_edge_update_mlps(node_hidden_dim, edge_hidden_dim, self.gnn_layer_num, activation, edge_metadata)


    def forward(self, node_dict, edge_attr_dict, edge_index_dict):

        for node, node_attr in node_dict.items():
            #node_dict[node] = self.onehot_transform(node_attr).squeeze(1)
            if node == "peptide":
                node_dict[node] = self.onehot_transform(node_attr).squeeze(1)
            elif node == "spectra":
                node_dict[node] = self.spectra_transform(node_attr)
            elif node == "protein":
                node_dict[node] = self.onehot_transform(node_attr).squeeze(1)

        if self.concat_states:
            concat_x_dict = defaultdict(list)

        for l, (conv_name, conv) in enumerate(zip(self.model_types, self.convs)):

            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                node_dict = conv(node_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
            elif conv_name == 'GCN':
                node_dict = conv(node_dict, edge_index_dict=edge_index_dict, edge_weight_dict=edge_attr_dict)
            else:
                node_dict = conv(node_dict, edge_index_dict)
            if self.concat_states:
                for node_type, x in node_dict:
                    concat_x_dict[node_type].append(x)

            #edge_attr_dict = self.update_edge_attr(node_dict, edge_attr_dict, edge_index_dict, self.edge_update_mlps[l])

        if self.concat_states:
            for node_type, x in concat_x_dict:
                node_dict[node_type] = torch.cat(concat_x_dict[node_type], 1)

        node_dict = self.update_node_attr(node_dict)
        return node_dict

    def update_node_attr(self, x_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_post_mlp[node_type](x)
        return x_dict

    def update_edge_attr(self, x_dict, edge_attr_dict, edge_index_dict, mlp):

        tmp_edge_attr_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            x_i = x_dict[src][edge_index[0],:]
            x_j = x_dict[dst][edge_index[1],:]
            tmp_edge_attr_dict[edge_type] = mlp["-".join(edge_type)](torch.cat((x_i,x_j,edge_attr_dict[edge_type]),dim=-1))
        return tmp_edge_attr_dict

    def build_edge_update_mlps(self, node_dim, edge_dim, gnn_layer_num, activation, edge_metadata):
        """
        Maintain a list of mlps for each edge_type. The first layer is different for each edge_type,
        the subsequent layers are identical for all edge_type.
        :param node_dim:
        :param edge_dim:
        :param gnn_layer_num:
        :param activation:
        :param edge_attr_dict:
        :return:
        """
        edge_update_mlps = torch.nn.ModuleList()

        edge_update_mlp_dict = torch.nn.ModuleDict()

        # Maintain a different layer for each input edge.
        for edge_type, (edge_input_dim, edge_mode) in edge_metadata.items():
            edge_update_mlp = torch.nn.Sequential(
                torch.nn.Linear(node_dim + node_dim + edge_input_dim, edge_dim),
                get_activation(activation),
            )
            edge_update_mlp_dict["-".join(edge_type)] = edge_update_mlp

        edge_update_mlps.append(edge_update_mlp_dict)

        # The rest of layers are the same for all edge types.
        for l in range(1, gnn_layer_num):
            edge_update_mlp = torch.nn.Sequential(
                torch.nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlp_dict = torch.nn.ModuleDict()
            for edge_type, _ in edge_metadata.items():
                edge_update_mlp_dict["-".join(edge_type)] = edge_update_mlp
            edge_update_mlps.append(edge_update_mlp_dict)
        return edge_update_mlps

    def build_convs(self, node_hidden_dim, edge_hidden_dim, edge_mode,
                     model_types, normalize_embs, activation, aggr, metadata):

        convs = torch.nn.ModuleList()

        edge_metadata = metadata["edge"]
        node_metadata = metadata["node"]

        node_attr_dim = list(node_metadata.values())[0]

        # conv_dict = {edge_type: self.build_conv_model(model_types[0], node_attr_dim, node_hidden_dim, edge_input_dim, \
        #                                   edge_mode, normalize_embs[0], activation, aggr) for edge_type, edge_input_dim in edge_metadata.items()}

        conv_dict = {edge_type: self.build_conv_model(model_types[0], node_hidden_dim, node_hidden_dim, edge_input_dim, \
                                          edge_mode, normalize_embs[0], activation, aggr) for edge_type, (edge_input_dim, edge_mode) in edge_metadata.items()}

        # base_conv = None
        # for index, (edge_type, conv) in enumerate(conv_dict.items()):
        #     if index == 0:
        #         base_conv = conv
        #     else:
        #         conv.agg_lin = base_conv.agg_lin   # make sure each layer share the same weights

        conv = HeteroConv(conv_dict, aggr = 'sum')

        convs.append(conv)
        for l in range(1, len(model_types)):

            # conv_dict = {edge_type: self.build_conv_model(model_types[0], node_hidden_dim, node_hidden_dim, edge_hidden_dim, \
            #                                               edge_mode, normalize_embs[0], activation, aggr) for edge_type, edge_input_dim in edge_metadata.items()}
            # conv_dict = {edge_type: self.build_conv_model(model_types[0], node_hidden_dim, node_hidden_dim, edge_input_dim, \
            #                                               edge_mode, normalize_embs[0], activation, aggr) for edge_type, edge_input_dim in edge_metadata.items()}
            conv_dict = {edge_type: self.build_conv_model(model_types[0], node_hidden_dim, node_hidden_dim, edge_input_dim, \
                                                          edge_mode, normalize_embs[0], activation, aggr) for edge_type, (edge_input_dim, edge_mode) in edge_metadata.items()}

            # base_conv = None
            # for index, (edge_type, conv) in enumerate(conv_dict.items()):
            #     if index == 0:
            #         base_conv = conv
            #     else:
            #         conv.agg_lin = base_conv.agg_lin    # make sure each layer share the same weights

            conv = HeteroConv(conv_dict, aggr='sum')

            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, \
                         edge_mode, normalize_emb, activation, aggr):
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim, node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim, node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim, node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim, node_out_dim, edge_dim, edge_mode)
        elif model_type == 'EGSAGE':
            return EGraphSage(node_in_dim, node_out_dim, edge_dim, activation, edge_mode, normalize_emb, aggr)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation, node_metadata):

        fn = torch.nn.ModuleDict()
        if 0 in hidden_dims:
            act = get_activation('none')
            for node_type, _ in node_metadata.items():
                fn[node_type] = act
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = torch.nn.Sequential(
                            torch.nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            torch.nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = torch.nn.Linear(input_dim, output_dim)
            layers.append(layer)

            final_layers = torch.nn.Sequential(*layers)
            for node_type, _ in node_metadata.items():
                fn[node_type] = final_layers

        return fn


class GNNStack_single(torch.nn.Module):

    def __init__(self,
                node_hidden_dim, edge_hidden_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs, aggr, metadata
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.metadata = metadata
        self.concat_states = concat_states
        self.model_types = model_types.split('_')
        self.gnn_layer_num = len(self.model_types)

        if normalize_embs is None:
            normalize_embs = [True, ] * len(self.model_types)
        else:
            normalize_embs = list(map(bool, map(int, normalize_embs.split('_'))))
        if node_post_mlp_hiddens is None:
            node_post_mlp_hiddens = []
        else:
            node_post_mlp_hiddens = list(map(int, node_post_mlp_hiddens.split('_')))
        print(self.model_types, normalize_embs, node_post_mlp_hiddens)

        edge_metadata = metadata["edge"]
        node_metadata = metadata["node"]

        self.protein_transform = torch.nn.Sequential(torch.nn.Linear(node_metadata["peptide"], node_hidden_dim), torch.nn.ReLU())
        self.peptide_transform = torch.nn.Sequential(torch.nn.Linear(node_metadata["peptide"], node_hidden_dim), torch.nn.ReLU())
        self.spectra_transform = torch.nn.Sequential(torch.nn.Linear(node_metadata["spectra"], node_hidden_dim), torch.nn.ReLU())
        self.onehot_transform = torch.nn.Embedding(2, embedding_dim=node_hidden_dim)#.to(device)

        # convs
        self.convs = self.build_convs(node_hidden_dim, 1, edge_mode,
                                    self.model_types, normalize_embs, activation, aggr, metadata)

        # post node update
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_hidden_dim * len(self.model_types)),
                                                          1, node_post_mlp_hiddens,
                                                          dropout, activation, node_metadata)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_hidden_dim, 1, node_post_mlp_hiddens, dropout,
                                                          activation, node_metadata)

        #self.edge_update_mlps = self.build_edge_update_mlps(node_hidden_dim, edge_hidden_dim, self.gnn_layer_num, activation)


    def forward(self, node_dict, edge_attr, edge_index):

        for node, node_attr in node_dict.items():
            if node == "peptide":
                node_dict[node] = self.onehot_transform(node_attr).squeeze(1)
            elif node == "spectra":
                node_dict[node] = self.spectra_transform(node_attr)
            elif node == "protein":
                node_dict[node] = self.onehot_transform(node_attr).squeeze(1)

        x = torch.cat([node_dict["protein"], node_dict["peptide"], node_dict["spectra"]], axis=0)

        if self.concat_states:
            concat_x = []
        for l,(conv_name,conv) in enumerate(zip(self.model_types, self.convs)):
            # self.check_input(x,edge_attr,edge_index)
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            if l != len(self.model_types) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.concat_states:
                concat_x.append(x)
            #edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
            #print(edge_attr.shape)
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        x = self.node_post_mlp(x)
        # self.check_input(x,edge_attr,edge_index)
        return x
        #return node_dict

    def update_node_attr(self, x_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_post_mlp[node_type](x)
        return x_dict

    def update_edge_attr(self, x_dict, edge_attr_dict, edge_index_dict, mlp):

        tmp_edge_attr_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            x_i = x_dict[src][edge_index[0],:]
            x_j = x_dict[dst][edge_index[1],:]
            tmp_edge_attr_dict[edge_type] = mlp["-".join(edge_type)](torch.cat((x_i,x_j,edge_attr_dict[edge_type]),dim=-1))
        return tmp_edge_attr_dict

    def build_edge_update_mlps(self, node_dim, edge_dim, gnn_layer_num, activation, edge_metadata):
        """
        Maintain a list of mlps for each edge_type. The first layer is different for each edge_type,
        the subsequent layers are identical for all edge_type.
        :param node_dim:
        :param edge_dim:
        :param gnn_layer_num:
        :param activation:
        :param edge_attr_dict:
        :return:
        """
        edge_update_mlps = torch.nn.ModuleList()

        edge_update_mlp_dict = torch.nn.ModuleDict()

        # Maintain a different layer for each input edge.
        for edge_type, (edge_input_dim, edge_mode) in edge_metadata.items():
            edge_update_mlp = torch.nn.Sequential(
                torch.nn.Linear(node_dim + node_dim + edge_input_dim, edge_dim),
                get_activation(activation),
            )
            edge_update_mlp_dict["-".join(edge_type)] = edge_update_mlp

        edge_update_mlps.append(edge_update_mlp_dict)

        # The rest of layers are the same for all edge types.
        for l in range(1, gnn_layer_num):
            edge_update_mlp = torch.nn.Sequential(
                torch.nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlp_dict = torch.nn.ModuleDict()
            for edge_type, _ in edge_metadata.items():
                edge_update_mlp_dict["-".join(edge_type)] = edge_update_mlp
            edge_update_mlps.append(edge_update_mlp_dict)
        return edge_update_mlps

    def build_convs(self, node_hidden_dim, edge_hidden_dim, edge_mode,
                     model_types, normalize_embs, activation, aggr, metadata):

        convs = torch.nn.ModuleList()

        conv = self.build_conv_model(model_types[0], node_hidden_dim, node_hidden_dim, edge_hidden_dim, \
                                          edge_mode, normalize_embs[0], activation, aggr)

        convs.append(conv)
        for l in range(1, len(model_types)):

            conv = self.build_conv_model(model_types[0], node_hidden_dim, node_hidden_dim, edge_hidden_dim, \
                                          edge_mode, normalize_embs[0], activation, aggr)

            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, \
                         edge_mode, normalize_emb, activation, aggr):
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim, node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim, node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim, node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim, node_out_dim, edge_dim, edge_mode)
        elif model_type == 'EGSAGE':
            return EGraphSage(node_in_dim, node_out_dim, edge_dim, activation, edge_mode, normalize_emb, aggr)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation, node_metadata):

        fn = torch.nn.ModuleDict()
        if 0 in hidden_dims:
            act = get_activation('none')
            return act
            # for node_type, _ in node_metadata.items():
            #     fn[node_type] = act
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = torch.nn.Sequential(
                            torch.nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            torch.nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = torch.nn.Linear(input_dim, output_dim)
            layers.append(layer)

            fn = torch.nn.Sequential(*layers)
            # for node_type, _ in node_metadata.items():
            #     fn[node_type] = final_layers

            return fn



class EGCNConv(MessagePassing):
    # form https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
    def __init__(self, in_channels, out_channels,
                 edge_channels, edge_mode,
                 improved=False, cached=False,
                 bias=True, **kwargs):
        super(EGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.edge_mode = edge_mode

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_mode == 0:
            self.attention_lin = torch.nn.Linear(2*out_channels+edge_channels, 1)
        elif self.edge_mode == 1:
            self.message_lin = torch.nn.Linear(2*out_channels+edge_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weight)
        zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_attr, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)


    def message(self, x_i, x_j, edge_attr, norm):
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * x_j
        elif self.edge_mode == 1:
            m_j = torch.cat((x_i, x_j, edge_attr),dim=-1)
            m_j = self.message_lin(m_j)
        return norm.view(-1, 1) * m_j

    def update(self, aggr_out, x):
        #print(aggr_out)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.edge_mode == 0:
            aggr_out = aggr_out + x
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class EGraphSage(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels,
                 edge_channels, activation, edge_mode,
                 normalize_emb,
                 aggr, bias=True):
        super(EGraphSage, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode

        if edge_mode == 0:
            self.message_lin = torch.nn.Linear(in_channels, out_channels)
            #self.attention_lin = torch.nn.Linear(2*in_channels+edge_channels, 1)
            self.attention_lin = torch.nn.Linear(edge_channels, 1)

        elif edge_mode == 6:
            #self.attention_lin = torch.nn.Linear(edge_channels, 1)
            self.message_lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        elif edge_mode == 7:
            self.message_lin = Linear(in_channels, out_channels, weight_initializer='glorot')
        elif edge_mode == 1: # Message layer, concat edge and node feature
            self.message_lin = Linear(edge_channels+in_channels, out_channels, weight_initializer='glorot')#torch.nn.Linear(in_channels+edge_channels, out_channels)
        elif edge_mode == 2:
            self.message_lin = torch.nn.Linear(2*in_channels+edge_channels, out_channels)
        elif edge_mode == 3:
            self.message_lin = torch.nn.Sequential(
                    torch.nn.Linear(2*in_channels+edge_channels, out_channels),
                    get_activation(activation),
                    torch.nn.Linear(out_channels, out_channels),
                    )
        elif edge_mode == 4:
            self.message_lin = torch.nn.Linear(in_channels, out_channels*edge_channels)
        elif edge_mode == 5:
            self.message_lin = torch.nn.Linear(2*in_channels, out_channels*edge_channels)
        elif edge_mode == 8:
            # self.sequence_encoder = EncoderRNN(3, 32, out_channels, 2)  # Here 32 is the hidden size, 2 is the num layers
            self.sequence_encoder = EncoderCNN(3, 8, out_channels)  # Here 4 is the window size

            self.message_lin = Linear(out_channels+in_channels, out_channels, weight_initializer='glorot')

        self.agg_lin = torch.nn.Linear(in_channels+out_channels, out_channels)

        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)
        self.normalize_emb = normalize_emb

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.message_lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_attr):
        #num_nodes = x.size(0)num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        num_nodes = max(x[0].size(0), x[1].size(0))
        #num_nodes = x[1].size(0)

        if self.edge_mode == 6:
            #edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr=edge_attr, fill_value=1, num_nodes=num_nodes)

            row, col = edge_index
            deg = scatter_add(edge_attr, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_attr = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]

            x = tuple([self.message_lin(x_) for x_ in x])
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            if self.bias is not None:
                out += self.bias
            return out#self.update_activation(out)
        # deg = degree(col, max(x[0].size(0), x[1].size(0)), dtype=x[0].dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        if self.edge_mode == 0:
            m_j = edge_attr * self.message_activation(self.message_lin(x_j))
            #attention = self.attention_lin(torch.cat((x_i, x_j, edge_attr),dim=-1))
            #attention = self.attention_lin(edge_attr)
            #m_j = attention * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 1:
            m_j = torch.cat((x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 2 or self.edge_mode == 3:
            m_j = torch.cat((x_i, x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 4:
            E = x_j.shape[0]
            w = self.message_lin(x_j)
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 5:
            E = x_j.shape[0]
            w = self.message_lin(torch.cat((x_i,x_j),dim=-1))
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 6:
            #assert edge_attr.shape[-1] == 1
            #attention = self.attention_lin(edge_attr)
            #m_j = attention * self.message_activation(self.message_lin(x_j))
            m_j = edge_attr * x_j# * norm.view(-1, 1)
            #m_j = attention * self.message_lin(x_j)
        elif self.edge_mode == 7:
            m_j = edge_attr * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 8:
            edge_feature = self.sequence_encoder(edge_attr)
            m_j = torch.cat((x_j, edge_feature), dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        #aggr_out = self.update_activation(aggr_out)
        if self.edge_mode != 6:
            aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x[1]),dim=-1)))
            if self.normalize_emb:
                aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
