import numpy as np


def get_proteins_by_fdr_forward(results, contaminate_proteins=None, fdr=0.01, indishtinguishable_group=None):

    shared_proteins = []
    if indishtinguishable_group is not None:
        for protein in contaminate_proteins:
            if protein in indishtinguishable_group:
                if np.array([(protein_ not in contaminate_proteins and not protein_.startswith("DECOY")) for protein_ in indishtinguishable_group[protein]]).any():
                    shared_proteins.append(protein)
       # contaminate_proteins = set(contaminate_proteins) - set(shared_proteins)


    tmp_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    true_proteins = []
    decoy_proteins = []
    for index, (protein, score) in enumerate(tmp_results):
        if contaminate_proteins is not None:
            if protein in contaminate_proteins:
                if protein not in shared_proteins:
                    decoy_proteins.append(protein)
            else:
                if not protein.startswith("DECOY"):
                    true_proteins.append(protein)
        else:
            if protein.startswith("DECOY"):
                decoy_proteins.append(protein)
            else:
                true_proteins.append(protein)

        if float(len(decoy_proteins) / (len(true_proteins)+len(decoy_proteins)+1e-20)) > fdr:
            break
    return true_proteins


def get_proteins_by_fdr(results,contaminate_proteins=None, fdr=0.01, max_len=5000, indishtinguishable_group=None):

    shared_proteins = []
    if indishtinguishable_group is not None:
        for protein in contaminate_proteins:
            if protein in indishtinguishable_group:
                if np.array([(protein_ not in contaminate_proteins and not protein_.startswith("DECOY")) for protein_ in indishtinguishable_group[protein]]).any():
                    shared_proteins.append(protein)
        contaminate_proteins = set(contaminate_proteins) - set(shared_proteins)

    tmp_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    if contaminate_proteins is not None and (np.array([protein.startswith("DECOY") for protein in contaminate_proteins])>0).all():
        contaminate_proteins = None

    for i in range(min(len(tmp_results), max_len), 0, -1):
        protein_pairs = tmp_results[:i]
        if contaminate_proteins is None:
            true_proteins = [protein for protein, score in protein_pairs if not protein.startswith("DECOY")]
            decoy_proteins = [protein for protein, score in protein_pairs if protein.startswith("DECOY")]

        else:
            # full_proteins = set([protein for protein, score in protein_pairs])
            # decoy_proteins = full_proteins.intersection(set(contaminate_proteins))
            proteins = [protein for protein, score in protein_pairs if not protein.startswith("DECOY")]
            proteins = set(proteins)
            true_proteins = proteins - set(contaminate_proteins) #

            decoy_proteins = proteins - true_proteins
            true_proteins = true_proteins - set(shared_proteins)

        if float(len(decoy_proteins) / (len(true_proteins) + len(decoy_proteins) + 1e-20)) > fdr:
            continue
        else:
            break

    return true_proteins

def get_proteins_by_decoy_fdr(results, fdr=0.01, max_len=5000):
    tmp_results = sorted(results.items(), key=lambda item: item[1], reverse=True) #TODO(m) this is the result

    for i in range(min(len(tmp_results), max_len), 0, -1):
        protein_pairs = tmp_results[:i]

        true_proteins = [protein for protein, score in protein_pairs if not protein.startswith("DECOY")]
        decoy_proteins = [protein for protein, score in protein_pairs if protein.startswith("DECOY")]

        if float(len(decoy_proteins) / (len(true_proteins) + len(decoy_proteins) + 1e-20)) > fdr:
            continue
        else:
            break

    return true_proteins

def get_proteins_by_decoy_fdr_test(results, fdr=0.01, max_len=5000):
    tmp_results = sorted(results.items(), key=lambda item: item[1], reverse=True) #TODO(m) this is the result

    for i in range(min(len(tmp_results), max_len), 0, -1):
        protein_pairs = tmp_results[:i]

        true_proteins = [protein for protein, score in protein_pairs if not protein.startswith("DECOY")]
        decoy_proteins = [protein for protein, score in protein_pairs if protein.startswith("DECOY")]

        if float(len(decoy_proteins) / (len(true_proteins) + len(decoy_proteins) + 1e-20)) > fdr:
            continue
        else:
            break

    return true_proteins + decoy_proteins