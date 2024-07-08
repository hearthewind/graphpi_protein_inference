#from datasets.dataset import Dataset_wSpectra as Dataset
from datasets.dataset import Dataset_wSpectra_hetero as Dataset
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from os.path import join
import pandas as pd


class IPRG2016TS(Dataset):

    def __init__(self, data_type, protein_label_type, prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy", pretrain_data_name="epifany"):
        dataset_folder = "/home/m/data1/Temp/Jupyter/two_species/iprg2016"

        self.psm_feature_path = join(f"{dataset_folder}/psm_features", f"result_{data_type.lower()}.json")
        self.protein_score_path = join(f"{dataset_folder}/{pretrain_data_name}_result", f"result_{data_type.lower()}.json")
        self.fido_protein_score_path = join(f"{dataset_folder}/fido_result", f"result_{data_type.lower()}.json")
        self.search_fasta = join(f"{dataset_folder}/decoy", f"decoy.fasta")

        self.data_type = data_type

        self.random_proteins = list(self.get_protein_sequences(join(f"{dataset_folder}/fasta", f"prest_1000_random.fasta")).keys())
        tmp = []
        for protein in self.random_proteins:
            tmp.append("CONTAMINATE_" + protein + "_TARGET")
        self.random_proteins = tmp

        self.a_proteins = list(self.get_protein_sequences(join(f"{dataset_folder}/fasta", f"prest_pool_a.fasta")).keys())
        tmp = []
        for protein in self.a_proteins:
            tmp.append("A_" + protein + "_TARGET")
        self.a_proteins = tmp

        self.b_proteins = list(self.get_protein_sequences(join(f"{dataset_folder}/fasta", f"prest_pool_b.fasta")).keys())
        tmp = []
        for protein in self.b_proteins:
            tmp.append("B_" + protein + "_TARGET")
        self.b_proteins = tmp

        self.second_proteins = list(self.get_protein_sequences(self.search_fasta).keys())
        tmp = []
        for protein in self.second_proteins:
            if protein.endswith("_SECOND"):
                if not protein.startswith("DECOY_"):
                    tmp.append(protein)
        self.second_proteins = tmp

        super().__init__("iPRG2016TS_"+data_type, protein_label_type, prior, train=train, prior_offset=prior_offset, filter_psm=filter_psm, output_type=output_type)
        pass

    def get_contaminate_proteins(self):
        if self.data_type == "A":
            contaminate_proteins = self.random_proteins + self.b_proteins + self.second_proteins
        elif self.data_type == "B":
            contaminate_proteins = self.random_proteins + self.a_proteins + self.second_proteins
        else:
            contaminate_proteins = self.random_proteins + self.second_proteins
        return contaminate_proteins

    def get_protein_labels_by_groundtruth(self): 
        if self.data_type == 'A':
            true_proteins = self.a_proteins
        elif self.data_type == 'B':
            true_proteins = self.b_proteins
        elif self.data_type == 'AB':
            true_proteins = self.a_proteins + self.b_proteins
        else:
            raise ValueError(f"No ground truth for iPRG2016_{self.data_type}")

        # neg_proteins = self.get_contaminate_proteins()
        # pos_proteins = true_proteins
        
        pos_proteins = []
        neg_proteins = []
        proteins = self.protein_map.keys()
        for protein in proteins:
            find_protein = False
            for true_protein in true_proteins:
                if protein.startswith(true_protein):
                    find_protein = True
                    break
            if find_protein is False:
                neg_proteins.append(protein)
            else:
                pos_proteins.append(protein)
        
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins
