#from datasets.dataset import Dataset_wSpectra as Dataset
from datasets.dataset import Dataset_wSpectra_hetero as Dataset
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from os.path import join
import pandas as pd

default_code = "3t3"
default_method = "mouse"
class Hela3T3(Dataset):

    def __init__(self, data_code=default_code, groundtruth_method=default_method, protein_label_type='benchmark', prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy", pretrain_data_name="epifany"):

        self.psm_feature_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"hela_3t3/psm_features", f"{data_code}.json")
        self.protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"hela_3t3/{pretrain_data_name}_result", f"{data_code}.json")
        self.search_fasta = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"hela_3t3", "database", "human_mouse_decoy.fasta")
        super().__init__(f"hela_3t3", protein_label_type, prior=prior, train=train, prior_offset=prior_offset, filter_psm=filter_psm, output_type=output_type)

        self.data_code = data_code
        self.groundtruth_method = groundtruth_method

        print('data_code', data_code)

        if groundtruth_method == 'human':
            self.true_proteins = list(self.get_protein_sequences(join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"hela_3t3/database", f"human_proteome.fasta")).keys())
        elif groundtruth_method == 'mouse':
            self.true_proteins = list(self.get_protein_sequences(join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"hela_3t3/database", f"mouse_proteome.fasta")).keys())
        else:
            raise NotImplementedError(f"groundtruth method {groundtruth_method} is not implemented")


    def get_protein_labels_by_groundtruth(self):
        true_proteins = set(self.true_proteins)
        pos_proteins = []
        neg_proteins = []
        proteins = self.protein_map.keys()
        for protein in proteins:
            if protein not in true_proteins:
                neg_proteins.append(protein)
            else:
                pos_proteins.append(protein)
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_contaminate_proteins(self):
        full_proteins = list(self.protein_map.keys())
        true_proteins = set(self.true_proteins)
        non_decoys = [protein for protein in full_proteins if not protein.startswith("DECOY")]
        return [protein for protein in non_decoys if protein not in true_proteins]

if __name__ == "__main__":
    data = Hela3T3(data_code=default_code, groundtruth_method=default_method, protein_label_type="decoy_sampling", filter_psm=False)
