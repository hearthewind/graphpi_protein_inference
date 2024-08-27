import numpy as np

def get_deeppep_feature(protein_seq: str, peptide_seq: str, pad_len: int):
    peptide_len = len(peptide_seq)
    protein_len = len(protein_seq)

    peptide_loc = protein_seq.find(peptide_seq)

    ret = np.zeros(pad_len)
    ret[0:protein_len] = 1
    if peptide_loc != -1:
        ret[peptide_loc:(peptide_loc + peptide_len)] = 2

    return ret