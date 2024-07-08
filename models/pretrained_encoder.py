from enum import Enum

from transformers import BertModel, BertTokenizer
import re
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
import os

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

cwd = os.getcwd()
with open(os.path.join(cwd, "uniprot_protein_rep.p"), 'rb') as f:
    uniprot_protein_rep = pickle.load(f)

cutoff = 3000

@Enum
class EncoderType:
    protein = 0
    peptide = 1

def helper(subsequence):
    new_sequence = ""
    for char in subsequence:
        new_sequence += char
        new_sequence += " "

    new_sequence = re.sub(r"[UZOB]", "X", new_sequence)
    encoded_input = tokenizer(new_sequence, return_tensors='pt')

    output = model(**encoded_input)

    ret = output.last_hidden_state[:, [0, -1], :].view(-1, 2048).detach().cpu().numpy()
    return ret


def get_representation(sequence):
    seq_len = len(sequence)
    if seq_len <= cutoff:
        return helper(sequence)
    else:
        return helper(sequence[:cutoff])


try:
    with open(os.path.join(cwd, "protein_rep.p"), 'rb') as f:
        cached_protein_rep = pickle.load(f)
except Exception as e:
    cached_protein_rep = {}

new_protein_counter = 0


def get_protein_rep_from_cache(name, sequence):
    if name in cached_protein_rep.keys():
        return cached_protein_rep[name].reshape(-1)
    elif name in uniprot_protein_rep.keys():
        return uniprot_protein_rep[name].reshape(-1)
    else:
        rep = get_representation(sequence)
        cached_protein_rep[name] = rep

        global new_protein_counter
        new_protein_counter += 1

        if new_protein_counter % 1000 == 0 or new_protein_counter < 1000:
            with open(os.path.join(cwd, "protein_rep.p"), 'wb') as f:
                pickle.dump(cached_protein_rep, f)
        return rep.reshape(-1)


try:
    with open(os.path.join(cwd, "peptide_rep.p"), 'rb') as f:
        cached_peptide_rep = pickle.load(f)
except Exception as e:
    cached_peptide_rep = {}

new_peptide_counter = 0


def get_peptide_rep_from_cache(sequence):
    if sequence in cached_peptide_rep.keys():
        return cached_peptide_rep[sequence].reshape(-1)
    else:
        new_peptide_rep = get_representation(sequence)
        cached_peptide_rep[sequence] = new_peptide_rep

        global new_peptide_counter
        new_peptide_counter += 1
        if new_peptide_counter % 1000 == 0 or new_peptide_counter < 1000:
            with open(os.path.join(cwd, "peptide_rep.p"), 'wb') as f:
                pickle.dump(cached_peptide_rep, f)

        return new_peptide_rep.reshape(-1)


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, type, device):
        super(Encoder, self).__init__()

        self.linear = nn.Linear(input_size, output_size - 1)

        if type == 'protein':
            self.type == EncoderType.protein
        elif type == 'peptide':
            self.type == EncoderType.peptide
        else:
            raise ValueError("Wrong type: {}".format(type))

        self.device = device

    def forward(self, input):

        ret = self.linear(input)
        ret = F.relu(ret)

        a, b = ret.shape

        if self.type == EncoderType.protein:
            vector = torch.zeros((a, 1)).to(self.device)
        else:
            vector = torch.ones((a, 1)).to(self.device)

        ret = torch.concat([ret, vector], axis=1)
        return ret