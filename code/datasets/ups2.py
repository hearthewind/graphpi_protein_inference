from datasets.dataset import Dataset_wSpectra_hetero as Dataset
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from os.path import join

class UPS2(Dataset):

    def __init__(self, protein_label_type, prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy", pretrain_data_name="epifany"):
        self.psm_feature_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, "ups2/psm_features", "result.json")
        self.protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"ups2/{pretrain_data_name}_result", "result.json")
        #self.search_fasta = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, "ups2", "decoy", "mask_all_decoy.fasta")
        self.search_fasta = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"ups2/database", "mask_all_decoy.fasta")

        super().__init__("ups2", protein_label_type, prior, train=train, prior_offset=prior_offset, filter_psm=filter_psm, output_type=output_type)

    def get_protein_labels_by_groundtruth(self):
        pos_proteins = []
        neg_proteins = []
        proteins = self.protein_map.keys()
        for protein in proteins:
            if protein.startswith("shuff") or protein.startswith("DECOY"):
                neg_proteins.append(protein)
            else:
                pos_proteins.append(protein)
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_contaminate_proteins(self):
        full_proteins = list(self.protein_map.keys())
        return [protein for protein in full_proteins if protein.startswith("shuff") and not protein.startswith("DECOY")]
