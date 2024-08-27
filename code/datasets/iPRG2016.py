from datasets.dataset import Dataset_wSpectra_hetero as Dataset
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from os.path import join
import pandas as pd


class IPRG2016(Dataset):

    def __init__(self, data_type, protein_label_type, prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy", pretrain_data_name="epifany"):
        dataset_folder = "iprg2016"
        # self.psm_feature_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"/iPRG2016/processed_data", f"psm_feature_{data_type}.json")
        # self.protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"/iPRG2016/processed_data", f"protein_score_{data_type}.json")
        self.psm_feature_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/psm_features", f"result_{data_type.lower()}.json")
        self.protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/{pretrain_data_name}_result", f"result_{data_type.lower()}.json")
        self.fido_protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/fido_result", f"result_{data_type.lower()}.json")
        self.search_fasta = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/database", f"decoy.fasta")

        self.data_type = data_type

        self.random_proteins = list(self.get_protein_sequences(join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/database", f"prest_1000_random.fasta")).keys())
        self.a_proteins = list(self.get_protein_sequences(join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/database", f"prest_pool_a.fasta")).keys())
        self.b_proteins = list(self.get_protein_sequences(join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"{dataset_folder}/database", f"prest_pool_b.fasta")).keys())
        super().__init__("iPRG2016_"+data_type, protein_label_type, prior, train=train, prior_offset=prior_offset, filter_psm=filter_psm, output_type=output_type)
        #self.get_neo4j_data(self.psm_features)
        pass

    def get_contaminate_proteins(self):
        if self.data_type == "A":
            contaminate_proteins = self.random_proteins + self.b_proteins
        elif self.data_type == "B":
            contaminate_proteins = self.random_proteins + self.a_proteins
        else:
            contaminate_proteins = self.random_proteins
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

    def get_customized_protein_name(self):
        contaminate_proteins = self.get_contaminate_proteins()
        custom_protein_map = dict()
        for protein in self.protein_map:
            if protein in contaminate_proteins:
                custom_protein_map[protein] = "CONTAMINATE_"+protein
            else:
                custom_protein_map[protein] = protein
        return custom_protein_map

    def get_neo4j_data(self, psm_features):
        general_mapping = []
        for psm in psm_features:
            proteins = psm["proteins"]
            spectra = psm["spectra"]
            peptide = psm["peptide"]
            pos_score = 1-psm["pep"]

            for protein in proteins:
                if protein in self.b_proteins:
                    protein = "B_"+protein
                elif protein in self.a_proteins:
                    protein = "A_"+protein
                elif protein in self.random_proteins:
                    protein = "CONTAMINATE_"+protein

                general_mapping.append({"spectra":spectra, "protein":protein, "peptide":peptide, "edge_weight":pos_score})

        general_mapping = pd.DataFrame(general_mapping)
        protein_peptide_mapping = general_mapping.groupby(["peptide", "protein"]).agg({"edge_weight":"max"}).reset_index()
        peptide_spectra_mapping = general_mapping[["spectra","peptide", "edge_weight"]].drop_duplicates()

        peptide_data = protein_peptide_mapping[["peptide","edge_weight"]].drop_duplicates()
        spectra_data = general_mapping[["spectra", "edge_weight"]].drop_duplicates()

        protein_peptide_mapping.to_csv(f"~/Downloads/protein_peptide_mapping_{self.data_type}.csv", index=False)
        peptide_spectra_mapping.to_csv(f"~/Downloads/peptide_spectra_mapping_{self.data_type}.csv", index=False)
        peptide_data.to_csv(f"~/Downloads/peptide_data_{self.data_type}.csv", index=False)
        spectra_data.to_csv(f"~/Downloads/spectra_data_{self.data_type}.csv", index=False)
