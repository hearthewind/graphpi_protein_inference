import torch

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'leaky':
        return torch.nn.LeakyReLU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'softplus':
        return torch.nn.Softplus()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError