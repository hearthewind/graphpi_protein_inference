#from datasets.dataset import Dataset_wSpectra as Dataset
from datasets.dataset import Dataset_wSpectra_hetero as Dataset
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from os.path import join
import pandas as pd


class PXD(Dataset):

    def __init__(self, data_code, protein_label_type, prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy", pretrain_data_name="epifany"):

        self.psm_feature_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"PXD{data_code}/psm_features", "result.json")
        self.protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"PXD{data_code}/{pretrain_data_name}_result", "result.json")
        self.search_fasta = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"uniprot", "uniprot_human_isoform_decoy.fasta")
        super().__init__(f"PXD{data_code}", protein_label_type, prior=prior, train=train, prior_offset=prior_offset, filter_psm=filter_psm, output_type=output_type)

    def get_protein_labels_by_groundtruth(self):
        pos_proteins = []
        neg_proteins = []
        proteins = self.protein_map.keys()
        for protein in proteins:
            if protein.startswith("DECOY"):
                neg_proteins.append(protein)
            else:
                pos_proteins.append(protein)
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_contaminate_proteins(self):
        full_proteins = list(self.protein_map.keys())
        return [protein for protein in full_proteins if protein.startswith("DECOY")]

    def get_neo4j_data(self, psm_features):
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
        peptide_spectra_mapping = general_mapping[["spectra", "peptide", "edge_weight"]].drop_duplicates()

        peptide_data = protein_peptide_mapping[["peptide", "edge_weight"]].drop_duplicates()
        spectra_data = general_mapping[["spectra", "edge_weight"]].drop_duplicates()

        protein_peptide_mapping.to_csv(f"~/Downloads/protein_peptide_mapping_{self.dataset}.csv", index=False)
        peptide_spectra_mapping.to_csv(f"~/Downloads/peptide_spectra_mapping_{self.dataset}.csv", index=False)
        peptide_data.to_csv(f"~/Downloads/peptide_data_{self.dataset}.csv", index=False)
        spectra_data.to_csv(f"~/Downloads/spectra_data_{self.dataset}.csv", index=False)

if __name__ == "__main__":
    data = PXD(data_code="005388", protein_label_type="decoy_sampling", filter_psm=False)
    data.get_neo4j_data(data.psm_features)
