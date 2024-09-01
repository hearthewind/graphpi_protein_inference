import logging

from datasets.deeppep import get_deeppep_feature
import abc
import json
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy as sc
from datasets.util import get_proteins_by_fdr, get_proteins_by_decoy_fdr, get_proteins_by_decoy_fdr_test
from configs import TEST_DATA
import random
import gc
import networkx as nx
from sortedcollections import OrderedSet
import re
from Bio import SeqIO

from copy import copy


class Dataset():

    def __init__(self, dataset, protein_label_type, prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy"):
        """

        :param dataset:
        :param protein_label_type:
        :param prior: Whether to use prior knowledge to change the weight of the protein-peptide mapping.
        """
        self.logger = logging.getLogger(__name__)
        self.dataset = dataset
        self.protein_label_type = protein_label_type
        self.prior = prior
        self.prior_offset = prior_offset
        self.train = train
        self.output = output_type

        if self.dataset.startswith("TestDataset"):
            pass
        else:
            self.protein_scores = self.extract_protein_score()
        self.psm_features = self.extract_psm_feature(filter=filter_psm)
        self.generate_id_map(self.psm_features)
        self.protein_seq_dict = self.get_protein_sequences(self.search_fasta)

    @abc.abstractmethod
    def preprocess(self):
        pass

    @abc.abstractmethod
    def get_contaminate_proteins(self):
        pass

    def extract_protein_score(self, protein_score_path=None):
        """
        Extract the protein scores from a benchmark algorithm (e.g. FIDO)
        :return:
        """
        file_name = protein_score_path if protein_score_path is not None else self.protein_score_path
        with open(file_name, "r") as f:
            json_data = json.load(f)

        protein_scores = {}
        for data in json_data:
            protein = data["accession"]
            score = data["best_search_engine_score[1]"]

            if data["opt_global_result_type"] == "protein_details":
                protein_scores[protein] = score

            # if data["opt_global_result_type"] == "indistinguishable_protein_group":
            #     members = data["ambiguity_members"]
            #     for member in members:
            #         protein_scores[member] = score

        return protein_scores

    def aggregate_protein_scores(self, protein_scores, type="atleast"):
        if type == "atleast":
            return 1-np.prod(1-np.array(protein_scores))
        elif type == "max":
            return np.max(protein_scores)
        elif type == "mean":
            return np.mean(protein_scores)
        elif type == "sum":
            return np.sum(protein_scores)

    def get_protein_sequences(self, file_name):
        # sequences = {}
        # with open(file_name, 'rb') as f:
        #     for name, sequence in parse_stream(f):
        #         name = name.decode('utf-8').split(' ')[0].split('\t')[0]
        #         sequence = sequence.decode('utf-8')
        #         sequences[name] = sequence
        # return sequences
        
        fasta_file_content = SeqIO.parse(open(file_name), 'fasta')
        proteins = {}
        for fasta in fasta_file_content:
            name = str(fasta.id)
            seq = str(fasta.seq)
            proteins[name] = seq
        return proteins

    def extract_psm_feature(self, filter=True):
        """
        Extract the psm features from the comet search.
        :return:
        """
        file_name = self.psm_feature_path
        with open(file_name, "r") as f:
            json_data = json.load(f)

        psm_features = []
        for data in json_data:
            if filter:
                if float(data["search_engine_score[1]"]) > 0.999:
                    continue
            psm_feature = {}
            psm_feature["peptide"] = data["sequence"]
            psm_feature["IonFrac"] = float(data["opt_global_COMET:IonFrac"])
            psm_feature["deltCn"] = float(data["opt_global_COMET:deltCn"])
            psm_feature["deltLCn"] = float(data["opt_global_COMET:deltLCn"])
            psm_feature["lnExpect"] = float(data["opt_global_COMET:lnExpect"])
            psm_feature["lnNumSP"] = float(data["opt_global_COMET:lnNumSP"])
            psm_feature["lnRankSP"] = float(data["opt_global_COMET:lnRankSP"])
            psm_feature["spScores"] = float(data["opt_global_MS:1002255"])
            psm_feature["xCorr"] = float(data["opt_global_MS:1002252"])
            psm_feature["proteins"] = data["accession"]
            #psm_feature["spectra"] = data["opt_global_spectrum_reference"]
            #psm_feature["spectra"] = data["spectra_ref"][0]
            #assert len(data["spectra_ref"]) == 1
            psm_feature["spectra"] = data["PSM_ID"]
            psm_feature["pep"] = float(data["search_engine_score[1]"])
            psm_feature["1001491"] = float(data["opt_global_MS:1001491"])
            psm_feature["1001492"] = float(data["opt_global_MS:1001492"])
            psm_feature["1001493"] = float(data["opt_global_MS:1001493"])
            psm_feature["1002252"] = float(data["opt_global_MS:1002252"])
            psm_feature["1002253"] = float(data["opt_global_MS:1002253"])
            psm_feature["1002254"] = float(data["opt_global_MS:1002254"])
            psm_feature["1002255"] = float(data["opt_global_MS:1002255"])
            psm_feature["1002256"] = float(data["opt_global_MS:1002256"])
            psm_feature["1002257"] = float(data["opt_global_MS:1002257"])
            psm_feature["1002258"] = float(data["opt_global_MS:1002258"])
            psm_feature["1002259"] = float(data["opt_global_MS:1002259"])


            ## add new features
            charge = int(data["charge"])
            psm_feature['charge_1'] = 0
            psm_feature['charge_2'] = 0
            psm_feature['charge_3'] = 0
            if charge == 1:
                psm_feature['charge_1'] = 1
            elif charge == 2:
                psm_feature['charge_2'] = 1
            elif charge == 3:
                psm_feature['charge_3'] = 1

            mz = float(data["exp_mass_to_charge"])
            mass = mz * charge - charge * 1.007276
            psm_feature['mass'] = mass

            theo_mass = float(data["calc_mass_to_charge"]) * charge - charge * 1.007276
            dm = mass - theo_mass
            absdm = abs(dm)
            psm_feature['dm'] = dm
            psm_feature['absdm'] = absdm

            psm_feature['peptide_length'] = len(data["sequence"])
            ## add the new features

            psm_features.append(psm_feature)
        return psm_features

    def get_protein_labels(self):

        # treat the decoy proteins as negative samples
        if self.protein_label_type == "decoy":
            return self.get_protein_labels_by_decoy()
        elif self.protein_label_type == "decoy_sampling":
            return self.get_protein_labels_by_resampling()
        # treat the decoy and contaminate proteins as negative samples
        elif self.protein_label_type == "groundtruth":
            return self.get_protein_labels_by_groundtruth()
        # simply treat all proteins as the negative samples (fully unsupervised)
        elif self.protein_label_type == "benchmark":
            return self.get_protein_labels_by_benchmark()
        elif self.protein_label_type == "test":
            return self.only_epifany()
        else:
            full_proteins = set(self.protein_map.keys())
            print(f"# pos proteins: 0, # neg proteins: {len(full_proteins)}")
            return [], full_proteins

    @abc.abstractmethod
    def get_protein_labels_by_groundtruth(self):
        pass

    def only_epifany(self):
        pos_proteins = get_proteins_by_decoy_fdr_test(self.protein_scores, fdr=0.05)

        full_proteins = self.protein_map.keys()

        nonpos_proteins = list(set(full_proteins) - set(pos_proteins))

        neg_proteins = nonpos_proteins
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def only_nonpos(self):
        pos_proteins = get_proteins_by_decoy_fdr(self.protein_scores, fdr=0.05)

        full_proteins = self.protein_map.keys()

        decoys = [protein for protein in full_proteins if protein.startswith("DECOY")]
        decoys = sorted(decoys, key=lambda protein:self.protein_scores[protein] if protein in self.protein_scores else 0)

        nonpos_proteins = list(set(full_proteins) - set(decoys) - set(pos_proteins))
        nonpos_proteins = sorted(nonpos_proteins, key=lambda protein: self.protein_scores[protein] if protein in self.protein_scores else 0)

        neg_proteins = nonpos_proteins
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_protein_labels_by_decoy(self):
        full_proteins = set(self.protein_map.keys())
        decoy_proteins = set([protein for protein in full_proteins if protein.startswith("DECOY")])
        neg_proteins = decoy_proteins
        pos_proteins = full_proteins - decoy_proteins
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_protein_labels_by_benchmark(self):
        pos_proteins = get_proteins_by_fdr(self.protein_scores, fdr=0.05)#, contaminate_proteins=self.get_contaminate_proteins())
        full_proteins = set(self.protein_map.keys())
        neg_proteins = full_proteins - set(pos_proteins)
        neg_proteins = sorted(neg_proteins, key=lambda protein:self.protein_scores[protein] if protein in self.protein_scores else 0)

        # neg_proteins = neg_proteins[:len(neg_proteins)-len(pos_proteins)]
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_protein_labels_by_resampling(self):
        pos_proteins = get_proteins_by_decoy_fdr(self.protein_scores, fdr=0.05)
        num_pos = len(pos_proteins)

        full_proteins = self.protein_map.keys()

        decoys = [protein for protein in full_proteins if protein.startswith("DECOY")]
        decoys = sorted(decoys, key=lambda protein:self.protein_scores[protein] if protein in self.protein_scores else 0)

        nonpos_proteins = list(set(full_proteins) - set(decoys) - set(pos_proteins))
        nonpos_proteins = sorted(nonpos_proteins, key=lambda protein: self.protein_scores[protein] if protein in self.protein_scores else 0)

        neg_proteins = decoys + nonpos_proteins #how do we to resample?

        highscore_decoys = decoys[-num_pos:]
        highscore_nonpos_proteins = nonpos_proteins[-num_pos:]
        lowscore_nonpos_proteins = nonpos_proteins[:num_pos]
        rest = decoys[:-num_pos] + nonpos_proteins[:-num_pos]

        if len(rest) < num_pos:
            random_chosen_rest = rest
        else:
            random_chosen_rest = random.sample(rest, num_pos)
        #neg_proteins = highscore_decoys + random_chosen_rest
        neg_proteins = highscore_decoys + highscore_nonpos_proteins + random_chosen_rest
        #neg_proteins = highscore_decoys + lowscore_nonpos_proteins + random_chosen_rest

        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def generate_peptide_overlap_scores(self, protein_peptide_mapping, peptide_score, offset=0.9):

        #protein_sequence = self.get_protein_sequences()
        data, i, j = protein_peptide_mapping.edge_weight.values, \
                     protein_peptide_mapping.peptide.values, protein_peptide_mapping.protein.values

        # calculate the peptide score as the maximum among the spectra scores.
        peptide_sum_score = {peptide_id:np.max(scores) for peptide_id, scores in peptide_score.items()}

        # calculate the protein score as the sum of the scores of the peptides.
        #pp_matrix = sc.sparse.coo_matrix((data, (i, j)), shape=(len(self.peptide_map)+len(self.protein_map), len(self.protein_map))).toarray().astype(float)
        pp_matrix = sc.sparse.coo_matrix((data, (i-len(self.protein_map), j)), shape=(len(self.peptide_map), len(self.protein_map))).tocsr().astype(float)#.toarray().astype(float)

        protein_len = len(self.protein_map)

        highscore_peptides_for_proteins = defaultdict(list)  #{protein: high score peptide list}
        peptide_position_for_protein = defaultdict(list)  #(protein, peptide): [(start, end)]
        for protein_id in self.protein_map.values():
            entries = pp_matrix[:, protein_id]
            #indexes = np.where(entries > 0)
            indexes = sc.sparse.find(entries)
            highscore_peptides = []
            for i in indexes[0]:
                if peptide_sum_score[i+protein_len] > offset:
                    highscore_peptides.append(i)
                    for match in re.finditer(self.peptide_sequence[i+protein_len], self.protein_sequence[protein_id]):
                        start, end = match.start(), match.end()
                        peptide_position_for_protein[(protein_id, i+protein_len)].append((start, end))

            highscore_peptides_for_proteins.append(highscore_peptides)

        overlap_score = []
        for _, peptide, protein in protein_peptide_mapping.values:
            if peptide_sum_score[peptide] > offset: # 本身就是高分peptide， 就直接赋值1.
                overlap_score.append(1)
            else:
                overlap = True
                # 但凡找到一个位置是该peptide不和任何一个高分有overlap的，就给分1.
                for match in re.finditer(self.peptide_sequence[peptide], self.protein_sequence[protein]):
                    start, end = match.start(), match.end()
                    for highscore_peptide in highscore_peptides_for_proteins[protein]:
                        positions = peptide_position_for_protein[highscore_peptide]
                        for position in positions:
                            if (start < position[0] and end < position[0]) or (start > position[1] and end > position[1]):
                                overlap = False
                                break
                        if overlap == False:
                            break
                    if overlap == False:
                        break
                overlap_score.append(1-int(overlap))

        return overlap_score

            #protein_scores[protein_id] = np.sum(np.array(scores) > offset)

    def process_degenerate(self, protein_peptide_mapping, peptide_score, offset=0.9):
        """
        For each peptide with shared proteins, assign the given peptide to the protein which\
        has the largest number of high score (>0.9) peptide siblings.
        :param protein_peptide_mapping:
        :param peptide_score:
        :return:
        """
        data, i, j = protein_peptide_mapping.edge_weight.values, \
                     protein_peptide_mapping.peptide.values, protein_peptide_mapping.protein.values

        # calculate the peptide score as the maximum among the spectra scores.
        peptide_sum_score = {peptide_id:np.max(scores) for peptide_id, scores in peptide_score.items()}

        # calculate the protein score as the sum of the scores of the peptides.
        #pp_matrix = sc.sparse.coo_matrix((data, (i, j)), shape=(len(self.peptide_map)+len(self.protein_map), len(self.protein_map))).toarray().astype(float)
        pp_matrix = sc.sparse.coo_matrix((data, (i-len(self.protein_map), j)), shape=(len(self.peptide_map), len(self.protein_map))).tocsr().astype(float)#.toarray().astype(float)
        #pp_matrix = sc.sparse.coo_matrix((data, (i-len(self.protein_map), j)), shape=(len(self.peptide_map), len(self.protein_map))).toarray().astype(float)

        protein_len = len(self.protein_map)

        protein_scores = defaultdict(list)
        for protein_id in self.protein_map.values():
            entries = pp_matrix[:, protein_id]
            #indexes = np.where(entries > 0)
            indexes = sc.sparse.find(entries)
            scores = []
            for i in indexes[0]:
                scores.append(peptide_sum_score[i+protein_len])
            protein_scores[protein_id] = np.sum(np.array(scores) > offset)
            #protein_scores[protein_id] = self.aggregate_protein_scores(scores, "mean")

        # protein_score_vector = np.zeros((pp_matrix.shape[1]))
        # for protein_id, score in protein_scores.items():
        #     protein_score_vector[protein_id] = score

        protein_score_vector = sc.sparse.csr_matrix(np.array(list(protein_scores.values()))).T

        # divide the weights of proteins for degenerate peptides.
        #protein_score_vector = np.exp(protein_score_vector)
        for peptide_id in self.peptide_map.values():
            entries = pp_matrix[peptide_id-protein_len, :]
            #entries = entries.toarray()[0]  # TODO dirty fix

            if (entries > 0).sum() > 1:
                # tmp = np.multiply(entries>0, protein_score_vector)
                # index = np.argwhere(tmp == np.amax(tmp))
                # assign = np.zeros_like(tmp)
                # assign[index] = 1
                # pp_matrix[peptide_id-protein_len, :] = pp_matrix[peptide_id-protein_len, :] * assign * 1/(entries>0).sum()

                tmp = (entries > 0).T.multiply(protein_score_vector)

                max_value = sc.sparse.csr_matrix.max(tmp)
                if max_value != 0:
                    index = sc.sparse.find(tmp == max_value)
                    #assign = sc.sparse.csr_matrix(np.zeros(tmp.shape))
                    assign = sc.sparse.csr_matrix((np.ones(len(index[0])), (index[0], index[1])), tmp.shape)
                    #assign[index[0]] = 1
                    pp_matrix[peptide_id-protein_len, :] = pp_matrix[peptide_id - protein_len, :].multiply(assign.T) * 1 / (entries > 0).sum()
                    #pp_matrix[peptide_id - protein_len, :] = pp_matrix[peptide_id - protein_len, :].multiply(assign.T) * 1 / len(assign.data)  # 分数不一样就对所有的高分进行一个平分。

                elif max_value == 0:
                    pp_matrix[peptide_id-protein_len, :] = pp_matrix[peptide_id - protein_len, :] * 1 / (entries > 0).sum()

        #pp_matrix = sc.sparse.coo_matrix(pp_matrix)
        pp_matrix = pp_matrix.tocoo()
        data,i,j = pp_matrix.data, pp_matrix.row, pp_matrix.col
        protein_peptide_mapping = pd.DataFrame({"edge_weight":data, "peptide":i+protein_len, "protein":j})
        return protein_peptide_mapping

    def get_connected_group(self, protein_peptide_mapping):
        G = nx.Graph()

        G.add_edges_from(protein_peptide_mapping[["peptide", "protein"]].values)

        components = nx.connected_components(G)
        components = list(components)

        groups = []
        #protein_ids = list(self.protein_map.values())
        protein_id_max = max(list(self.protein_map.values()))
        for component in components:
            group = []
            for node in component:
                if node <= protein_id_max:
                    group.append(node)
            group = set(group)
            if len(group)>1:
                if group not in groups:
                    groups.append(group)
        return groups

    def get_indistinguishable_group(self, protein_peptide_mapping, peptide_score, threshold=0.1):

        data, i, j = protein_peptide_mapping.edge_weight.values, \
                     protein_peptide_mapping.peptide.values, protein_peptide_mapping.protein.values

        # calculate the peptide score as the maximum among the spectra scores.
        peptide_sum_score = {peptide_id: np.max(scores) for peptide_id, scores in peptide_score.items()}

        # calculate the protein score as the sum of the scores of the peptides.
        pp_matrix = sc.sparse.coo_matrix((data, (i-len(self.protein_map), j)), shape=(
        len(self.peptide_map), len(self.protein_map))).tocsr()#.toarray().astype(float)

        protein_similarity_matrix = pp_matrix.transpose()*pp_matrix
        protein_pair_indexes = sc.sparse.find(protein_similarity_matrix)[:2]

        protein_scores = defaultdict(list)
        protein_len = len(self.protein_map)
        for protein_id in self.protein_map.values():
            entries = pp_matrix[:, protein_id]
            indexes = sc.sparse.find(entries)
            scores = []
            for i in indexes[0]:
                i = protein_len+i
                scores.append(peptide_sum_score[i])
            protein_scores[protein_id] = self.aggregate_protein_scores(scores, "sum")

        indistinguishable_groups = defaultdict(list)
        for protein_pair in zip(*protein_pair_indexes):
            if protein_pair[0] == protein_pair[1]:
                continue

            protein_1, protein_2 = protein_pair[0], protein_pair[1]

            if np.abs(protein_scores[protein_1]-protein_scores[protein_2])< threshold:
                indistinguishable_groups[protein_1].append(protein_2)
                indistinguishable_groups[protein_2].append(protein_1)

        for protein, protein_pairs in indistinguishable_groups.items():
            indistinguishable_groups[protein] = list(set(protein_pairs))

        return indistinguishable_groups

    def get_protein_with_multiple_peptides(self, protein_peptide_mapping):

        def func(sub):
            num_high_score_peptides = len(sub[sub.edge_weight > 0.9]) # Have two high score peptides
            return num_high_score_peptides >= 2

        num_peptides_per_proteins = protein_peptide_mapping.sort_values('edge_weight', ascending=True).groupby(["protein"]).apply(lambda sub: func(sub))
        protein_with_multiple_peptides = num_peptides_per_proteins[num_peptides_per_proteins == True].reset_index().protein.values
        return protein_with_multiple_peptides

    def add_bayesian_prior(self, protein_peptide_mapping, alpha=0.1): # TODO(m) change this function
        """
        1/n*(1-alpha^n)
        :param protein_peptide_mapping:
        :return:
        """

        num_protein_per_peptide = dict(protein_peptide_mapping.groupby("peptide").agg({"protein":"nunique"}).reset_index().values)

        new_mapping = []
        for protein, peptide in protein_peptide_mapping[["protein", "peptide"]].values:
            n = num_protein_per_peptide[peptide]
            edge_weight = 1/n * (1-alpha**n)
            new_mapping.append({"protein":protein, "peptide":peptide, "edge_weight": edge_weight})

        new_protein_peptide_mapping = pd.DataFrame(new_mapping)
        return new_protein_peptide_mapping

    def process_one_hit(self, protein_peptide_mapping, peptide_score, offset=0.9):
        """
        For each peptide, if it's connected protein has less than or equal to 1 high-score peptides, then their protein-peptide weight / 0.5.
        :param protein_peptide_mapping:
        :param peptide_score:
        :return:
        """
        data, i, j = protein_peptide_mapping.edge_weight.values, \
                     protein_peptide_mapping.peptide.values, protein_peptide_mapping.protein.values

        # calculate the peptide score as the maximum among the spectra scores.
        peptide_sum_score = {peptide_id:np.max(scores) for peptide_id, scores in peptide_score.items()}

        # calculate the protein score as the sum of the scores of the peptides.
        pp_matrix = sc.sparse.coo_matrix((data, (i, j)), shape=(len(self.peptide_map)+len(self.protein_map), len(self.protein_map))).toarray().astype(float)
        #pp_matrix = sc.sparse.coo_matrix((data, (i, j)), shape=(len(self.peptide_map)+len(self.protein_map), len(self.protein_map))).tocsr()
        protein_scores = defaultdict(list)
        for protein_id in self.protein_map.values():
            entries = pp_matrix[:, protein_id]
            indexes = np.where(entries > 0)
            #indexes = sc.sparse.find(entries)
            scores = []
            for i in indexes[0]:
                scores.append(peptide_sum_score[i])
            protein_scores[protein_id] = np.sum(np.array(scores) > offset) #self.aggregate_protein_scores(scores, "sum")

        protein_score_vector = np.zeros((len(self.protein_map)))
        for protein_id, score in protein_scores.items():
            protein_score_vector[protein_id] = score

        for peptide_id in self.peptide_map.values():
            # if peptide_id == self.peptide_map["WWTCFVKR"]: # protein_id: 2576
            #     print(peptide_id)
            entries = pp_matrix[peptide_id, :]
            #protein_score = (protein_score_vector)*entries
            #pp_matrix[peptide_id, :] = entries * np.where(protein_score_vector >= 2, 1, 0.5) # optimal
            pp_matrix[peptide_id, :] = entries * np.where(protein_score_vector >= 2, 1, 0.5)

            # protein_score = (protein_score_vector - peptide_sum_score[peptide_id]>offset)*entries
            # pp_matrix[peptide_id, :] = entries * np.where(protein_score >= 1, 1, 0.5)

            # If the rest of protein_score is larger than 1, meaning that there are shared peptides for the same protein of the peptide.
            #pp_matrix[peptide_id, :] = entries * (protein_score >= 1) #1/(1+np.exp(-10*(protein_score-0.6)))#1/(1+np.exp(-4*protein_score))#(protein_score > 0)
            #pp_matrix[peptide_id, :] = sc.sparse.csr_matrix(entries * np.where(protein_score >= 1, 1, 0.5))

        pp_matrix = sc.sparse.coo_matrix(pp_matrix)
        data,i,j = pp_matrix.data, pp_matrix.row, pp_matrix.col
        protein_peptide_mapping = pd.DataFrame({"edge_weight":data, "peptide":i, "protein":j})
        return protein_peptide_mapping


class Dataset_wSpectra(Dataset):

    def preprocess(self):

        edge_mapping, edge_weights = self.generate_edge_mapping(self.psm_features)
        protein_features, peptide_features, spectra_features = self.process_node_features(self.psm_features)

        protein_features = torch.tensor(protein_features, dtype=torch.float)
        peptide_features = torch.tensor(peptide_features, dtype=torch.float)
        spectra_features = torch.tensor(spectra_features, dtype=torch.float)
        self.node_features = {"protein_features":protein_features, "peptide_features":peptide_features, "spectra_features":spectra_features}
        self.edge = torch.LongTensor(edge_mapping)
        self.edge_weights = torch.FloatTensor(edge_weights)

        if self.train:
            pos_proteins, neg_proteins = self.get_protein_labels()
            self.pos_proteins = [self.id_map[protein] for protein in pos_proteins if protein in self.id_map]
            self.neg_proteins = [self.id_map[protein] for protein in neg_proteins if protein in self.id_map]
            self.proteins = self.pos_proteins+self.neg_proteins

            if self.output == "soft_entropy":
                self.y = torch.FloatTensor(np.zeros(len(self.id_map)))
                reverse_id_map = {value:key for key,value in self.id_map.items()}
                for protein in self.proteins: #TODO
                    if reverse_id_map[protein].startswith("DECOY"):
                        self.y[protein] = 0.0
                    else:
                        try:
                            self.y[protein] = self.protein_scores[reverse_id_map[protein]]
                        except KeyError as e:
                            self.y[protein] = 0.0
                    # try:
                    #     self.y[protein] = self.protein_scores[reverse_id_map[protein]]
                    # except KeyError as e:
                    #     self.y[protein] = 0.0
                            
            elif self.output == "cross_entropy":
                self.y = torch.LongTensor(np.zeros(len(self.id_map)))
                for id_ in self.pos_proteins:
                    self.y[id_] = 1
        else:
            self.proteins = list(self.protein_map.values())
            contaminate_proteins = self.get_contaminate_proteins()
            pos_proteins = [protein_id for protein, protein_id in self.protein_map.items() if protein not in contaminate_proteins and not protein.startswith("DECOY")]

            if self.output == "soft_entropy":

                self.y = torch.FloatTensor(np.zeros(len(self.id_map)))
                reverse_id_map = {value:key for key,value in self.id_map.items()}

                for protein in self.proteins: #TODO
                    if protein.startswith("DECOY"):
                        self.y[protein] = 0.0
                    else:
                        try:
                            self.y[protein] = self.protein_scores[reverse_id_map[protein]]
                        except KeyError as e:
                            self.y[protein] = 0.0
            else:
                self.y = torch.LongTensor(np.zeros(len(self.id_map)))
                for id_ in pos_proteins:
                    self.y[id_] = 1

        #del self.psm_features
        gc.collect()

    def generate_id_map(self, psm_features):
        """
        Convert the protein, spectra and peptide to unique ids.
        :param psm_features:
        :return:
        """
        peptides, proteins, spectras = [], [], []
        for psm in psm_features:
            proteins.extend(psm["proteins"])
            peptides.append(psm["peptide"])
            spectras.append(psm["spectra"])
        peptides,proteins,spectras = set(peptides),set(proteins),set(spectras)
        self.protein_map = dict(zip(proteins, range(len(proteins))))
        self.peptide_map = dict(zip(peptides, range(len(proteins), len(proteins)+len(peptides))))
        self.spectra_map = dict(zip(spectras, range(len(proteins)+len(peptides), len(proteins)+len(peptides)+len(spectras))))
        self.id_map = dict(zip(list(proteins)+list(peptides)+list(spectras), range(len(proteins)+len(peptides)+len(spectras))))

    def generate_edge_mapping(self, psm_features):
        """
        Generate the tri-partite graph.
        :param psm_features:
        :return:
        """

        general_mapping = []
        for psm in psm_features:
            proteins = psm["proteins"]
            spectra = psm["spectra"]
            peptide = psm["peptide"]
            pos_score = 1-psm["pep"]

            for protein in proteins:
                general_mapping.append({"spectra":spectra, "protein":protein, "peptide":peptide, "edge_weight":pos_score})

        general_mapping = pd.DataFrame(general_mapping)
        protein_peptide_mapping = general_mapping.groupby(["peptide", "protein"]).agg({"edge_weight":"max"}).reset_index()

        #peptide_spectra_mapping = general_mapping.groupby(["spectra","peptide"]).agg({"edge_weight":"max"}).reset_index()
        peptide_spectra_mapping = general_mapping.sort_values('edge_weight', ascending=True).groupby(['peptide', 'spectra']).tail(1)[["spectra", "peptide","edge_weight"]]

        #peptide_spectra_mapping = general_mapping[["spectra", "peptide", "edge_weight"]].drop_duplicates()

        protein_peptide_mapping.protein = protein_peptide_mapping.protein.apply(lambda x: self.protein_map[x])
        protein_peptide_mapping.peptide = protein_peptide_mapping.peptide.apply(lambda x: self.peptide_map[x])
        peptide_spectra_mapping.peptide = peptide_spectra_mapping.peptide.apply(lambda x: self.peptide_map[x])
        peptide_spectra_mapping.spectra = peptide_spectra_mapping.spectra.apply(lambda x: self.spectra_map[x])

        peptide_score = dict(peptide_spectra_mapping.groupby("peptide")["edge_weight"].apply(lambda x: x.tolist()).reset_index().values)

        # peptide_spectra_mapping.columns = ["start_id", "end_id", "edge_weight"]
        # protein_peptide_mapping.columns = ["start_id", "end_id", "edge_weight"]


        if self.dataset in TEST_DATA:# and self.train is False:
            self.indistinguishable_groups = self.get_indistinguishable_group(protein_peptide_mapping, peptide_score, threshold=0.001)
        self.protein_with_multiple_peptides = self.get_protein_with_multiple_peptides(protein_peptide_mapping)
        self.degenerate_groups = self.get_connected_group(protein_peptide_mapping)


        protein_peptide_mapping.edge_weight = 1

        if self.prior:
            #protein_peptide_mapping = self.process_one_hit(protein_peptide_mapping=protein_peptide_mapping, peptide_score=peptide_score)
            protein_peptide_mapping = self.process_degenerate(peptide_score=peptide_score, protein_peptide_mapping=protein_peptide_mapping, offset=self.prior_offset)
            #protein_peptide_mapping = self.add_bayesian_prior(protein_peptide_mapping, alpha=0.9)
             # #protein_peptide_mapping = self.add_bayesian_prior(protein_peptide_mapping, alpha=0.1)
            #protein_peptide_mapping = self.add_bayesian_prior(protein_peptide_mapping, alpha=0.01)


        protein_peptide_mapping.rename(columns = {"peptide":"start_id", "protein":"end_id"}, inplace=True)
        peptide_spectra_mapping.rename(columns = {"spectra":"start_id", "peptide":"end_id"}, inplace=True)
        reverse_protein_peptide_mapping = protein_peptide_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})#[["end_id", "start_id", "edge_weight"]]
        reverse_peptide_spectra_mapping = peptide_spectra_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})

        protein_peptide_mapping = pd.concat([protein_peptide_mapping, reverse_protein_peptide_mapping], axis=0)
        peptide_spectra_mapping = pd.concat([peptide_spectra_mapping, reverse_peptide_spectra_mapping], axis=0)
        edge_mapping = pd.concat([peptide_spectra_mapping, protein_peptide_mapping], axis=0)
        return edge_mapping[["start_id", "end_id"]].values, edge_mapping[["edge_weight"]].values

    def generate_edge_mapping_with_one_hit_proteins(self):

        general_mapping = []
        for psm in self.psm_features:
            proteins = psm["proteins"]
            spectra = psm["spectra"]
            peptide = psm["peptide"]
            pos_score = 1-psm["pep"]

            for protein in proteins:
                general_mapping.append({"spectra":self.spectra_map[spectra], "protein":self.protein_map[protein], "peptide":self.peptide_map[peptide], "edge_weight":pos_score})

        general_mapping = pd.DataFrame(general_mapping)

        kept_proteins = self.protein_with_multiple_peptides

        general_mapping = general_mapping[general_mapping["protein"].isin(kept_proteins)]

        protein_peptide_mapping = general_mapping.groupby(["peptide", "protein"]).agg({"edge_weight": "max"}).reset_index()

        def func(sub):
            high_score_peptides = sub[sub.edge_weight > 0.9].peptide.tolist()
            peptide = random.sample(high_score_peptides, k=1)
            low_score_peptides = sub["peptide"].head(len(sub) - len(high_score_peptides)).tolist()
            return sub[sub.peptide.isin(peptide + low_score_peptides)]

        protein_peptide_mapping = protein_peptide_mapping.sort_values('edge_weight', ascending=True).groupby(["protein"]).apply(lambda sub: func(sub))

        peptide_spectra_mapping = general_mapping.sort_values('edge_weight', ascending=True).groupby(['peptide', 'spectra']).tail(1)[["spectra", "peptide","edge_weight"]]

        peptide_score = dict(peptide_spectra_mapping.groupby("peptide")["edge_weight"].apply(lambda x: x.tolist()).reset_index().values)

        # Since there is no proteins with shared peptides.
        if self.prior:
            #protein_peptide_mapping = self.process_one_hit(protein_peptide_mapping=protein_peptide_mapping, peptide_score=peptide_score)
            protein_peptide_mapping = self.process_degenerate(peptide_score=peptide_score, protein_peptide_mapping=protein_peptide_mapping, offset=self.prior_offset)


        protein_peptide_mapping.rename(columns = {"peptide":"start_id", "protein":"end_id"}, inplace=True)
        peptide_spectra_mapping.rename(columns = {"spectra":"start_id", "peptide":"end_id"}, inplace=True)
        reverse_protein_peptide_mapping = protein_peptide_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})#[["end_id", "start_id", "edge_weight"]]
        reverse_peptide_spectra_mapping = peptide_spectra_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})

        protein_peptide_mapping = pd.concat([protein_peptide_mapping, reverse_protein_peptide_mapping], axis=0)
        peptide_spectra_mapping = pd.concat([peptide_spectra_mapping, reverse_peptide_spectra_mapping], axis=0)
        edge_mapping = pd.concat([peptide_spectra_mapping, protein_peptide_mapping], axis=0)
        self.one_hit_edge = torch.LongTensor(edge_mapping[["start_id", "end_id"]].values)
        self.one_hit_edge_weights = torch.FloatTensor(edge_mapping[["edge_weight"]].values)
        self.one_hit_proteins = kept_proteins

    def generate_edge_mapping_without_shared_proteins(self):

        general_mapping = []
        for psm in self.psm_features:
            proteins = psm["proteins"]
            spectra = psm["spectra"]
            peptide = psm["peptide"]
            pos_score = 1-psm["pep"]

            for protein in proteins:
                general_mapping.append({"spectra":self.spectra_map[spectra], "protein":self.protein_map[protein], "peptide":self.peptide_map[peptide], "edge_weight":pos_score})

        general_mapping = pd.DataFrame(general_mapping)

        kept_proteins = []
        for group in self.degenerate_groups:
            #if len(group) > 1:
            protein = int(np.random.choice(group, size=1)[0])
            #rest_proteins = list(group).remove(protein)
            kept_proteins.append(protein)

        general_mapping = general_mapping[general_mapping["protein"].isin(kept_proteins)]

        protein_peptide_mapping = general_mapping.groupby(["peptide", "protein"]).agg({"edge_weight": "max"}).reset_index()

        peptide_spectra_mapping = general_mapping.sort_values('edge_weight', ascending=True).groupby(['peptide', 'spectra']).tail(1)[["spectra", "peptide","edge_weight"]]

        peptide_score = dict(peptide_spectra_mapping.groupby("peptide")["edge_weight"].apply(lambda x: x.tolist()).reset_index().values)

        # Since there is no proteins with shared peptides.
        if self.prior:
            #protein_peptide_mapping = self.process_one_hit(protein_peptide_mapping=protein_peptide_mapping, peptide_score=peptide_score)
            protein_peptide_mapping = self.process_degenerate(peptide_score=peptide_score, protein_peptide_mapping=protein_peptide_mapping)


        protein_peptide_mapping.rename(columns = {"peptide":"start_id", "protein":"end_id"}, inplace=True)
        peptide_spectra_mapping.rename(columns = {"spectra":"start_id", "peptide":"end_id"}, inplace=True)
        reverse_protein_peptide_mapping = protein_peptide_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})#[["end_id", "start_id", "edge_weight"]]
        reverse_peptide_spectra_mapping = peptide_spectra_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})

        protein_peptide_mapping = pd.concat([protein_peptide_mapping, reverse_protein_peptide_mapping], axis=0)
        peptide_spectra_mapping = pd.concat([peptide_spectra_mapping, reverse_peptide_spectra_mapping], axis=0)
        edge_mapping = pd.concat([peptide_spectra_mapping, protein_peptide_mapping], axis=0)
        self.single_edge = torch.LongTensor(edge_mapping[["start_id", "end_id"]].values)
        self.single_edge_weights = torch.FloatTensor(edge_mapping[["edge_weight"]].values)
        self.single_proteins = kept_proteins

    def process_node_features(self, psm_features):
        spectra_features = {}
        spectra_scores = {}
        #peptide_scores = {}
        #peptide_features = {}

        feature_names = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores",\
                        "xCorr", "pep", ]# "1001491", "1001492", "1001493", "1002252", "1002253", "1002254",\
                        # "1002255", "1002256", "1002257", "1002258", "1002259"]

        feature_names = ["pep"]
        # ["pep", "xCorr", "1002253", "lnExpect", "1002256"]
        # {"IonFrac": (188, 10), "deltaCn": (188, 9), "deltaLCn": (186, 12), "lnExpect": (184, 20), "suspect_logits": (121, 14), "spScores":(189, 12),\
        # "xCorr": (190, 19), "1001491": (187, 12), "lnNumSP": (190, 13), "1001492": (185, 18), "1001493": (187, 19), "1002252": (187, 17), "1002253": (186, 18),\
        # "1002254":(190, 12), "1002255": (187, 14), "1002256": (186, 18), "1002257":(184, 1), "1002258": (184, 19), "1002259":(190, 10)}

        for psm in psm_features:
            #spectra_features[psm["spectra"]] = [psm["pep"]]
            spectra_features[psm["spectra"]] = [psm[feature] for feature in feature_names]

            # spectra_features[psm["spectra"]] = [psm["IonFrac"], psm["deltCn"], psm["deltLCn"], \
            #                                         psm["lnExpect"], psm["lnNumSP"],psm["lnRankSP"],\
            #                                         psm["spScores"],psm["xCorr"], psm["pep"],\
            #                    psm["1001491"], psm["1001492"], psm["1001493"], \
            #                    psm["1002252"], psm["1002253"], psm["1002254"], psm["1002255"],\
            #                    psm["1002256"], psm["1002257"], psm["1002258"], psm["1002259"]]
            # if psm["spectra"] not in spectra_scores or (1-psm["pep"]) > spectra_scores[psm["spectra"]]:
            #     spectra_features[psm["spectra"]] = [psm["IonFrac"], psm["deltCn"], psm["deltLCn"], \
            #                                         psm["lnExpect"], psm["lnNumSP"],psm["lnRankSP"],\
            #                                         psm["spScores"],psm["xCorr"], psm["pep"],\
            #                    psm["1001491"], psm["1001492"], psm["1001493"], \
            #                    psm["1002252"], psm["1002253"], psm["1002254"], psm["1002255"],\
            #                    psm["1002256"], psm["1002257"], psm["1002258"], psm["1002259"]]
            #     spectra_scores[psm["spectra"]] = 1-psm["pep"]

            # if psm["peptide"] not in peptide_scores or (1-psm["pep"]) > peptide_scores[psm["peptide"]]:
            #     peptide_features[psm["peptide"]] = [psm["pep"]]
            #     peptide_scores[psm["peptide"]] = 1-psm["pep"]


        spectra_feature = []
        for name in self.spectra_map:
            spectra_feature.append(spectra_features[name])
        scaler = StandardScaler()
        spectra_feature = scaler.fit_transform(np.array(spectra_feature))

        # one_hot feature for protein
        protein_feature = np.zeros((len(self.protein_map), 1))
        #one_hot feature for peptide
        peptide_feature = np.ones((len(self.peptide_map), 1))
        # protein_feature = np.zeros((len(self.protein_map), 2))
        # protein_feature[:,0] = 1
        # #one_hot feature for peptide
        # peptide_feature = np.zeros((len(self.peptide_map), 2))
        # peptide_feature[:, 1] = 1
        # peptide_feature_add = []
        # for name in self.peptide_map:
        #     peptide_feature_add.append(peptide_features[name])
        # peptide_feature_add = np.array(peptide_feature_add)
        # # scaler = StandardScaler()
        # # peptide_feature_add = scaler.fit_transform(np.array(peptide_feature_add))
        # peptide_feature = np.concatenate([peptide_feature, peptide_feature_add], axis=1)

        return protein_feature, peptide_feature, spectra_feature


class Dataset_wSpectra_hetero(Dataset):

    def preprocess(self):

        self.edge_dict, self.edge_attr_dict = self.generate_edge_mapping(self.psm_features)
        self.node_features = self.generate_node_features(self.psm_features)

        if self.train:

            if self.output == "cross_entropy":
                pos_proteins, neg_proteins = self.get_protein_labels()
                self.pos_proteins = [self.id_map[protein] for protein in pos_proteins if protein in self.id_map]
                self.neg_proteins = [self.id_map[protein] for protein in neg_proteins if protein in self.id_map]
                self.proteins = self.pos_proteins + self.neg_proteins
            else:
                self.proteins = list(self.protein_map.values())

            self.update_label()
            #protein_sample_weights = self.get_protein_sample_weights(self.pos_proteins, self.neg_proteins)
            #self.node_weight = [protein_sample_weights[protein_id] if protein_id in protein_sample_weights else 0 for
            #                    protein, protein_id in self.id_map.items()]
        else:
            self.proteins = list(self.protein_map.values())
            contaminate_proteins = self.get_contaminate_proteins()
            pos_proteins = [protein_id for protein, protein_id in self.protein_map.items() if
                            protein not in contaminate_proteins and not protein.startswith("DECOY")]

            self.y = torch.LongTensor(np.zeros(len(self.protein_map)))
            for id_ in pos_proteins:
                self.y[id_] = 1

        gc.collect()

    def update_selftrain(self, args, new_true_proteins, new_false_proteins, new_result):
        if self.output == "cross_entropy":
            percentage = args.percentage

            len_new_true = int(len(new_true_proteins) * percentage)
            len_new_false = int(len(new_false_proteins) * percentage)
            new_pos_proteins = [self.id_map[protein] for protein in new_true_proteins if protein in self.id_map][:len_new_true]
            new_neg_proteins = [self.id_map[protein] for protein in new_false_proteins if protein in self.id_map][-len_new_false:]

            if args.concat_old:
                self.pos_proteins = list(set(self.pos_proteins + new_pos_proteins))
                self.neg_proteins = list(set(self.neg_proteins + new_neg_proteins))
            else:
                self.pos_proteins = new_pos_proteins
                self.neg_proteins = new_neg_proteins

            self.proteins = self.pos_proteins + self.neg_proteins
        else: # soft_entropy
            self.protein_scores = new_result
            self.proteins = list(self.protein_map.values())
        self.update_label()

    def reset_label(self):
        if self.output == "cross_entropy":
            pos_proteins, neg_proteins = self.get_protein_labels()
            self.pos_proteins = [self.id_map[protein] for protein in pos_proteins if protein in self.id_map]
            self.neg_proteins = [self.id_map[protein] for protein in neg_proteins if protein in self.id_map]
            self.proteins = self.pos_proteins + self.neg_proteins
        else:
            self.proteins = list(self.protein_map.values())

        self.update_label()

    def update_label(self):
        if self.output == "soft_entropy":
            self.y = torch.FloatTensor(np.zeros(len(self.protein_map)))
            reverse_id_map = {value: key for key, value in self.protein_map.items()}
            for protein in self.proteins:  # TODO
                if reverse_id_map[protein].startswith("DECOY"):
                    self.y[protein] = 0.0

                else:
                    if protein in reverse_id_map:
                        try:
                            self.y[protein] = self.protein_scores[reverse_id_map[protein]]
                        except KeyError as e:
                            self.y[protein] = 0.0
        elif self.output == "cross_entropy":
            self.y = torch.LongTensor(np.zeros(len(self.protein_map)))
            for id_ in self.pos_proteins:
                self.y[id_] = 1

    def generate_id_map(self, psm_features):
        """
        Convert the protein, spectra and peptide to unique ids.
        :param psm_features:
        :return:
        """
        peptides, proteins, spectras = [], [], []
        for psm in psm_features:
            proteins.extend(psm["proteins"])
            peptides.append(psm["peptide"])
            spectras.append(psm["spectra"])
        peptides,proteins,spectras = OrderedSet(peptides),OrderedSet(proteins),OrderedSet(spectras)
        self.protein_map = dict(zip(proteins, range(len(proteins))))
        self.peptide_map = dict(zip(peptides, range(len(peptides))))
        self.spectra_map = dict(zip(spectras, range(len(spectras))))
        self.id_map = dict(
            zip(list(proteins) + list(peptides) + list(spectras), range(len(proteins) + len(peptides) + len(spectras))))
        self.reverse_id_map = {value: key for key, value in self.id_map.items()}
        self.id_to_protein_map = dict(zip(range(len(proteins)), range(len(proteins))))
        self.id_to_peptide_map = dict(zip(range(len(proteins), len(proteins) + len(peptides)), range(len(peptides))))
        self.id_to_spectra_map = dict(
            zip(range(len(proteins) + len(peptides), len(proteins) + len(peptides) + len(spectras)),
                range(len(spectras))))

    def generate_edge_mapping(self, psm_features):
        """
        Generate the tri-partite graph.
        :param psm_features:
        :return:
        """

        general_mapping = []
        for psm in psm_features:
            proteins = psm["proteins"]
            spectra = psm["spectra"]
            peptide = psm["peptide"]
            pos_score = 1 - psm["pep"]

            for protein in proteins:
                general_mapping.append(
                    {"spectra": spectra, "protein": protein, "peptide": peptide, "edge_weight": pos_score})

        general_mapping = pd.DataFrame(general_mapping)
        protein_peptide_mapping = general_mapping.groupby(["peptide", "protein"]).agg(
            {"edge_weight": "max"}).reset_index()
        protein_peptide_mapping.edge_weight = 1

        old_protein_peptide_mapping = copy(protein_peptide_mapping)

        peptide_spectra_mapping = general_mapping.groupby(["spectra", "peptide"]).agg(
            {"edge_weight": "max"}).reset_index()

        # to calculate the prior, need unique id for peptides+proteins.
        protein_peptide_mapping.protein = protein_peptide_mapping.protein.apply(lambda x: self.id_map[x])
        protein_peptide_mapping.peptide = protein_peptide_mapping.peptide.apply(lambda x: self.id_map[x])
        peptide_spectra_mapping.peptide = peptide_spectra_mapping.peptide.apply(lambda x: self.id_map[x])
        peptide_spectra_mapping.spectra = peptide_spectra_mapping.spectra.apply(lambda x: self.id_map[x])

        peptide_score = dict(
            peptide_spectra_mapping.groupby("peptide")["edge_weight"].apply(lambda x: x.tolist()).reset_index().values)

        if self.dataset in TEST_DATA:  # and self.train is False:
            self.indistinguishable_groups = self.get_indistinguishable_group(protein_peptide_mapping, peptide_score,
                                                                             threshold=0.001)
        else: #TODO(m): This is only for self-train, need to be removed.
            self.indistinguishable_groups = self.get_indistinguishable_group(protein_peptide_mapping, peptide_score,
                                                                             threshold=0.001)

        self.protein_with_multiple_peptides = self.get_protein_with_multiple_peptides(protein_peptide_mapping)
        self.degenerate_groups = self.get_connected_group(protein_peptide_mapping)

        protein_peptide_mapping.edge_weight = 1

        if self.prior:
            protein_peptide_mapping = self.process_degenerate(peptide_score=peptide_score,
                                                              protein_peptide_mapping=protein_peptide_mapping,
                                                              offset=self.prior_offset)

        # for heterogeneous GNN, need id for each node start with 0.
        protein_peptide_mapping.protein = protein_peptide_mapping.protein.apply(lambda x: self.id_to_protein_map[x])
        protein_peptide_mapping.peptide = protein_peptide_mapping.peptide.apply(lambda x: self.id_to_peptide_map[x])
        peptide_spectra_mapping.peptide = peptide_spectra_mapping.peptide.apply(lambda x: self.id_to_peptide_map[x])
        peptide_spectra_mapping.spectra = peptide_spectra_mapping.spectra.apply(lambda x: self.id_to_spectra_map[x])

        edge_dict = {}

        edge_dict[("peptide", "rev_contain", "protein")] = torch.tensor(
            protein_peptide_mapping[["peptide", "protein"]].values.T, dtype=torch.long)
        edge_dict[("protein", "contain", "peptide")] = torch.tensor(
            protein_peptide_mapping[["protein", "peptide"]].values.T, dtype=torch.long)
        edge_dict[("spectra", "rev_contain", "peptide")] = torch.tensor(
            peptide_spectra_mapping[["spectra", "peptide"]].values.T, dtype=torch.long)
        edge_dict[("peptide", "contain", "spectra")] = torch.tensor(
            peptide_spectra_mapping[["peptide", "spectra"]].values.T, dtype=torch.long)

        edge_attr_dict = self.generate_edge_attr(psm_features, peptide_spectra_mapping, protein_peptide_mapping, old_protein_peptide_mapping)

        return edge_dict, edge_attr_dict

    def generate_edge_attr(self, psm_features, peptide_spectra_mapping, protein_peptide_mapping, old_protein_peptide_mapping):
        """
        Generate the edge attr for protein-peptide and peptide-spectra
        :param psm_features:
        :param peptide_spectra_mapping:
        :param protein_peptide_mapping:
        :return:
        """
        edge_attr_dict = {}

        peptide_spectra_attr = []
        # raw_feature_cols = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores",\
        #                "xCorr", "pep", "1001491", "1001492", "1001493", "1002252", "1002253", "1002254", "1002255", "1002256",\
        #                "1002257", "1002258", "1002259"]
        raw_feature_cols = ["pep"]
        feature_cols = ["spectra", "peptide"] + raw_feature_cols

        for psm in psm_features:
            peptide_spectra_attr.append({col: psm[col] for col in feature_cols})

        # peptide_spectra_attr = pd.DataFrame(peptide_spectra_attr).drop_duplicates(["peptide", "spectra"])
        tmp = pd.DataFrame(peptide_spectra_attr)
        peptide_spectra_attr = tmp.sort_values('pep', ascending=False).groupby(['peptide', 'spectra']).tail(1)
        peptide_spectra_attr.peptide = peptide_spectra_attr.peptide.apply(lambda x: self.peptide_map[x])
        peptide_spectra_attr.spectra = peptide_spectra_attr.spectra.apply(lambda x: self.spectra_map[x])
        peptide_spectra_attr = peptide_spectra_mapping[["peptide", "spectra"]].merge(peptide_spectra_attr).drop(
            columns=["peptide", "spectra"])
        peptide_spectra_attr["pep"] = 1 - peptide_spectra_attr["pep"]  # make the score is 1-pep

        if not (len(raw_feature_cols) == 1 and "pep" in raw_feature_cols):
            scaler = StandardScaler()
            peptide_spectra_attr = scaler.fit_transform(peptide_spectra_attr)
        else:
            peptide_spectra_attr = peptide_spectra_attr.values

        edge_attr_dict[("spectra", "rev_contain", "peptide")] = torch.tensor(peptide_spectra_attr, dtype=torch.float)
        edge_attr_dict[("peptide", "contain", "spectra")] = torch.tensor(peptide_spectra_attr, dtype=torch.float)

        if self.use_deeppep:
            protein_peptide_feature = torch.tensor(self.generate_protein_peptide_feature(old_protein_peptide_mapping), \
                                                   dtype=torch.int)
        else:
            protein_peptide_feature = torch.tensor(protein_peptide_mapping.edge_weight.values.reshape(-1, 1), \
                                                   dtype=torch.float)
        edge_attr_dict[("peptide", "rev_contain", "protein")] = protein_peptide_feature
        edge_attr_dict[("protein", "contain", "peptide")] = protein_peptide_feature

        return edge_attr_dict

    def generate_protein_peptide_feature(self, old_protein_peptide_mapping):
        ret = []
        for peptide_seq, protein_accession, _ in old_protein_peptide_mapping.values:
            try:
                protein_seq = self.protein_seq_dict[protein_accession]
            except KeyError as e:
                print(self.protein_seq_dict.keys())
                raise(e)
            feature = get_deeppep_feature(protein_seq, peptide_seq, 1000)
            ret.append(feature)
        return np.array(ret)

    def generate_node_features(self, psm_features):

        spectra_features = {}
        feature_names = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores",\
                        "xCorr", "pep", "charge_1", "charge_2", "charge_3", "mass", "dm", "absdm", "peptide_length"]

        feature_names = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores", \
                         "xCorr", "pep"]

        # feature_names = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores", \
        #                  "xCorr", "pep", "1001491", "1001492", "1001493", "1002252", "1002253", "1002254",\
        #                  "1002255", "1002256", "1002257", "1002258", "1002259"]
        # feature_names = ["pep"]
        #print(feature_names)

        for psm in psm_features:
            spectra_features[psm["spectra"]] = [psm[feature] for feature in feature_names]


        spectra_feature = []
        for name in self.spectra_map:
            spectra_feature.append(spectra_features[name])
        scaler = StandardScaler()
        spectra_feature = scaler.fit_transform(np.array(spectra_feature))

        protein_feature = np.ones((len(self.protein_map), 1)) * 0
        peptide_feature = np.ones((len(self.peptide_map), 1)) * 1
        #spectra_feature = np.ones((len(self.spectra_map), 1)) * 2

        node_feature_dict = {"peptide": torch.tensor(peptide_feature, dtype=torch.long), \
                             "protein": torch.tensor(protein_feature, dtype=torch.long), \
                             "spectra": torch.tensor(spectra_feature, dtype=torch.float)}

        return node_feature_dict

    def process_degenerate(self, protein_peptide_mapping, peptide_score, offset=0.9):
        """
        For each peptide with shared proteins, assign the given peptide to the protein which\
        has the largest number of high score (>0.9) peptide siblings.
        :param protein_peptide_mapping:
        :param peptide_score:
        :return:
        """
        data, i, j = protein_peptide_mapping.edge_weight.values, \
                     protein_peptide_mapping.peptide.values, protein_peptide_mapping.protein.values

        # calculate the peptide score as the maximum among the spectra scores.

        # calculate the protein score as the sum of the scores of the peptides.
        #pp_matrix = sc.sparse.coo_matrix((data, (i, j)), shape=(len(self.peptide_map)+len(self.protein_map), len(self.protein_map))).toarray().astype(float)
        pp_matrix = sc.sparse.coo_matrix((data, (i-len(self.protein_map), j)), shape=(len(self.peptide_map), len(self.protein_map))).tocsr().astype(float)#.toarray().astype(float)
        #pp_matrix = sc.sparse.coo_matrix((data, (i-len(self.protein_map), j)), shape=(len(self.peptide_map), len(self.protein_map))).toarray().astype(float)

        protein_len = len(self.protein_map)
        peptide_sum_score = {peptide_id-protein_len: np.max(scores) for peptide_id, scores in peptide_score.items()}

        protein_scores = defaultdict(list)
        for protein_id in self.protein_map.values():
            entries = pp_matrix[:, protein_id]
            #indexes = np.where(entries > 0)
            indexes = sc.sparse.find(entries)
            scores = []
            for i in indexes[0]:
                scores.append(peptide_sum_score[i])
            protein_scores[protein_id] = np.sum(np.array(scores) > offset)
            #protein_scores[protein_id] = self.aggregate_protein_scores(scores, "mean")

        # protein_score_vector = np.zeros((pp_matrix.shape[1]))
        # for protein_id, score in protein_scores.items():
        #     protein_score_vector[protein_id] = score

        protein_score_vector = sc.sparse.csr_matrix(np.array(list(protein_scores.values()))).T

        # divide the weights of proteins for degenerate peptides.
        #protein_score_vector = np.exp(protein_score_vector)
        for peptide_id in self.peptide_map.values():
            entries = pp_matrix[peptide_id, :]
            #entries = entries.toarray()[0]  # TODO dirty fix

            if (entries > 0).sum() > 1:
                # tmp = np.multiply(entries>0, protein_score_vector)
                # index = np.argwhere(tmp == np.amax(tmp))
                # assign = np.zeros_like(tmp)
                # assign[index] = 1
                # pp_matrix[peptide_id-protein_len, :] = pp_matrix[peptide_id-protein_len, :] * assign * 1/(entries>0).sum()

                tmp = (entries > 0).T.multiply(protein_score_vector)

                max_value = sc.sparse.csr_matrix.max(tmp)
                if max_value != 0:
                    index = sc.sparse.find(tmp == max_value)
                    #assign = sc.sparse.csr_matrix(np.zeros(tmp.shape))
                    assign = sc.sparse.csr_matrix((np.ones(len(index[0])), (index[0], index[1])), tmp.shape)
                    #assign[index[0]] = 1
                    pp_matrix[peptide_id, :] = pp_matrix[peptide_id, :].multiply(assign.T) * 1 / (entries > 0).sum()
                    #pp_matrix[peptide_id - protein_len, :] = pp_matrix[peptide_id - protein_len, :].multiply(assign.T) * 1 / len(assign.data)  # 分数不一样就对所有的高分进行一个平分。

                elif max_value == 0:
                    pp_matrix[peptide_id, :] = pp_matrix[peptide_id, :] * 1 / (entries > 0).sum()

        #pp_matrix = sc.sparse.coo_matrix(pp_matrix)
        pp_matrix = pp_matrix.tocoo()
        data,i,j = pp_matrix.data, pp_matrix.row, pp_matrix.col
        protein_peptide_mapping = pd.DataFrame({"edge_weight":data, "peptide":i+protein_len, "protein":j})
        return protein_peptide_mapping




class Dataset_wSpectra_hetero_single(Dataset):

    def preprocess(self):

        self.edge_index, self.edge_attr = self.generate_edge_mapping(self.psm_features)
        self.node_features = self.generate_node_features(self.psm_features)

        if self.train:
            pos_proteins, neg_proteins = self.get_protein_labels()
            self.pos_proteins = [self.id_map[protein] for protein in pos_proteins if protein in self.id_map]
            self.neg_proteins = [self.id_map[protein] for protein in neg_proteins if protein in self.id_map]
            self.proteins = self.pos_proteins+self.neg_proteins

            self.update_label()
            #protein_sample_weights = self.get_protein_sample_weights(self.pos_proteins, self.neg_proteins)
            #self.node_weight = [protein_sample_weights[protein_id] if protein_id in protein_sample_weights else 0 for protein, protein_id in self.id_map.items()]
        else:
            self.proteins = list(self.protein_map.values())
            contaminate_proteins = self.get_contaminate_proteins()
            pos_proteins = [protein_id for protein, protein_id in self.protein_map.items() if protein not in contaminate_proteins and not protein.startswith("DECOY")]

            self.y = torch.LongTensor(np.zeros(len(self.id_map)))
            for id_ in pos_proteins:
                self.y[id_] = 1

        #del self.psm_features
        gc.collect()


    def update_label(self):
        if self.output == "soft_entropy":
            self.y = torch.FloatTensor(np.zeros(len(self.id_map)))
            reverse_id_map = {value: key for key, value in self.id_map.items()}
            for protein in self.proteins:  # TODO
                if reverse_id_map[protein].startswith("DECOY"):
                    self.y[protein] = 0.0

                else:
                    if protein in reverse_id_map:
                        try:
                            self.y[protein] = self.protein_scores[reverse_id_map[protein]]
                        except KeyError as e:
                            self.y[protein] = 0.0
        elif self.output == "cross_entropy":
            self.y = torch.LongTensor(np.zeros(len(self.id_map)))
            for id_ in self.pos_proteins:
                self.y[id_] = 1


    def generate_id_map(self, psm_features):
        """
        Convert the protein, spectra and peptide to unique ids.
        :param psm_features:
        :return:
        """
        peptides, proteins, spectras = [], [], []
        for psm in psm_features:
            proteins.extend(psm["proteins"])
            peptides.append(psm["peptide"])
            spectras.append(psm["spectra"])
        peptides,proteins,spectras = OrderedSet(peptides),OrderedSet(proteins),OrderedSet(spectras)
        self.protein_map = dict(zip(proteins, range(len(proteins))))
        self.peptide_map = dict(zip(peptides, range(len(proteins), len(proteins)+len(peptides))))
        self.spectra_map = dict(zip(spectras, range(len(proteins)+len(peptides), len(proteins)+len(peptides)+len(spectras))))
        self.id_map = dict(zip(list(proteins)+list(peptides)+list(spectras), range(len(proteins)+len(peptides)+len(spectras))))
        self.reverse_id_map = {value:key for key,value in self.id_map.items()}

    def generate_edge_mapping(self, psm_features):
        """
        Generate the tri-partite graph.
        :param psm_features:
        :return:
        """

        general_mapping = []
        for psm in psm_features:
            proteins = psm["proteins"]
            spectra = psm["spectra"]
            peptide = psm["peptide"]
            pos_score = 1-psm["pep"]

            for protein in proteins:
                general_mapping.append({"spectra":spectra, "protein":protein, "peptide":peptide, "edge_weight":pos_score})

        general_mapping = pd.DataFrame(general_mapping)
        protein_peptide_mapping = general_mapping.groupby(["peptide", "protein"]).agg({"edge_weight":"max"}).reset_index()

        #peptide_spectra_mapping = general_mapping.groupby(["spectra","peptide"]).agg({"edge_weight":"max"}).reset_index()
        peptide_spectra_mapping = general_mapping.sort_values('edge_weight', ascending=True).groupby(['peptide', 'spectra']).tail(1)[["spectra", "peptide","edge_weight"]]

        #peptide_spectra_mapping = general_mapping[["spectra", "peptide", "edge_weight"]].drop_duplicates()

        protein_peptide_mapping.protein = protein_peptide_mapping.protein.apply(lambda x: self.protein_map[x])
        protein_peptide_mapping.peptide = protein_peptide_mapping.peptide.apply(lambda x: self.peptide_map[x])
        peptide_spectra_mapping.peptide = peptide_spectra_mapping.peptide.apply(lambda x: self.peptide_map[x])
        peptide_spectra_mapping.spectra = peptide_spectra_mapping.spectra.apply(lambda x: self.spectra_map[x])

        peptide_score = dict(
            peptide_spectra_mapping.groupby("peptide")["edge_weight"].apply(lambda x: x.tolist()).reset_index().values)

        if self.dataset in TEST_DATA:  # and self.train is False:
            self.indistinguishable_groups = self.get_indistinguishable_group(protein_peptide_mapping, peptide_score,
                                                                             threshold=0.001)
        self.protein_with_multiple_peptides = self.get_protein_with_multiple_peptides(protein_peptide_mapping)
        self.degenerate_groups = self.get_connected_group(protein_peptide_mapping)

        protein_peptide_mapping.edge_weight = 1

        if self.prior:
            protein_peptide_mapping = self.process_degenerate(peptide_score=peptide_score,
                                                              protein_peptide_mapping=protein_peptide_mapping,
                                                              offset=self.prior_offset)

        #edge_attr = self.generate_edge_attr(psm_features, peptide_spectra_mapping, protein_peptide_mapping)

        protein_peptide_mapping.rename(columns = {"peptide":"start_id", "protein":"end_id"}, inplace=True)
        peptide_spectra_mapping.rename(columns = {"spectra":"start_id", "peptide":"end_id"}, inplace=True)
        reverse_protein_peptide_mapping = protein_peptide_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})#[["end_id", "start_id", "edge_weight"]]
        reverse_peptide_spectra_mapping = peptide_spectra_mapping.rename(columns={"end_id":"start_id", "start_id":"end_id"})

        protein_peptide_mapping = pd.concat([protein_peptide_mapping, reverse_protein_peptide_mapping], axis=0)
        peptide_spectra_mapping = pd.concat([peptide_spectra_mapping, reverse_peptide_spectra_mapping], axis=0)
        edge_mapping = pd.concat([peptide_spectra_mapping, protein_peptide_mapping], axis=0)

        edge_attr = torch.tensor(edge_mapping.edge_weight.values.reshape((-1, 1)),  dtype=torch.float)

        edge_mapping = torch.tensor(edge_mapping[["start_id","end_id"]].values.T,  dtype=torch.long)

        return edge_mapping, edge_attr

    def generate_edge_attr(self, psm_features, peptide_spectra_mapping, protein_peptide_mapping):
        """
        Generate the edge attr for protein-peptide and peptide-spectra
        :param psm_features:
        :param peptide_spectra_mapping:
        :param protein_peptide_mapping:
        :return:
        """

        peptide_spectra_attr = []
        raw_feature_cols = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores", \
                            "xCorr", "pep", "1001491", "1001492", "1001493", "1002252", "1002253", "1002254", "1002255",
                            "1002256", \
                            "1002257", "1002258", "1002259"]
        raw_feature_cols = ["pep"]
        feature_cols = ["spectra", "peptide"] + raw_feature_cols

        for psm in psm_features:
            peptide_spectra_attr.append({col: psm[col] for col in feature_cols})

        # peptide_spectra_attr = pd.DataFrame(peptide_spectra_attr).drop_duplicates(["peptide", "spectra"])
        tmp = pd.DataFrame(peptide_spectra_attr)
        peptide_spectra_attr = tmp.sort_values('pep', ascending=False).groupby(['peptide', 'spectra']).tail(1)
        peptide_spectra_attr.peptide = peptide_spectra_attr.peptide.apply(lambda x: self.peptide_map[x])
        peptide_spectra_attr.spectra = peptide_spectra_attr.spectra.apply(lambda x: self.spectra_map[x])
        peptide_spectra_attr = peptide_spectra_mapping[["peptide", "spectra"]].merge(peptide_spectra_attr).drop(
            columns=["peptide", "spectra"])
        peptide_spectra_attr["pep"] = 1 - peptide_spectra_attr["pep"]  # make the score is 1-pep

        if not (len(raw_feature_cols) == 1 and "pep" in raw_feature_cols):
            scaler = StandardScaler()
            peptide_spectra_attr = scaler.fit_transform(peptide_spectra_attr)
        else:
            peptide_spectra_attr = peptide_spectra_attr.values

        protein_peptide_attr = torch.tensor(
            protein_peptide_mapping.edge_weight.values.reshape((-1, 1)), dtype=torch.float)

        peptide_spectra_attr = torch.tensor(peptide_spectra_attr, dtype=torch.float)
        edge_attr = torch.cat([protein_peptide_attr, protein_peptide_attr, peptide_spectra_attr, peptide_spectra_attr], axis=0)

        return edge_attr

    def generate_node_features(self, psm_features):

        spectra_features = {}
        feature_names = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores",\
                        "xCorr", "pep", ]

        # feature_names = ["IonFrac", "deltCn", "deltLCn", "lnExpect", "lnNumSP", "lnRankSP", "spScores", \
        #                  "xCorr", "pep", "1001491", "1001492", "1001493", "1002252", "1002253", "1002254",\
        #                  "1002255", "1002256", "1002257", "1002258", "1002259"]
        #feature_names = ["pep"]
        # print(feature_names)

        for psm in psm_features:
            spectra_features[psm["spectra"]] = [psm[feature] for feature in feature_names]


        spectra_feature = []
        for name in self.spectra_map:
            spectra_feature.append(spectra_features[name])
        scaler = StandardScaler()
        spectra_feature = scaler.fit_transform(np.array(spectra_feature))

        protein_feature = np.ones((len(self.protein_map), 1)) * 0
        peptide_feature = np.ones((len(self.peptide_map), 1)) * 1
        #spectra_feature = np.ones((len(self.spectra_map), 1)) * 2

        node_feature_dict = {"peptide": torch.tensor(peptide_feature, dtype=torch.long), \
                             "protein": torch.tensor(protein_feature, dtype=torch.long), \
                             "spectra": torch.tensor(spectra_feature, dtype=torch.float)}

        return node_feature_dict