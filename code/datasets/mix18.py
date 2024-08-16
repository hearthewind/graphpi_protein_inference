from datasets.dataset import Dataset_wSpectra_hetero as Dataset
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from os.path import join

class Mix18(Dataset):

    def __init__(self, protein_label_type, prior=True, prior_offset=0.9, train=True, filter_psm=True, output_type="cross_entropy", pretrain_data_name="epifany"):
        #self.ground_truth_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, "yeast", "raw_data", "yeast_5MSdatasets_in2ormore.lst.txt")
        self.ground_truth_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, "18mix", "database", "18mix.fasta")
        self.psm_feature_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, "18mix/psm_features", "result.json")
        self.protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, f"18mix/{pretrain_data_name}_result", "result.json")
        self.search_fasta = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR, "18mix/database", "decoy.fasta")
        
        self.true_proteins = list(self.get_protein_sequences(self.ground_truth_path).keys())
        #print(f"lenght of true proteins {len(self.true_proteins)}")
        super().__init__("18mix", protein_label_type, prior, train=train, prior_offset=prior_offset, filter_psm=filter_psm, output_type=output_type)

        #self.custom_protein_map = self.get_customized_protein_name()

    def get_protein_labels_by_groundtruth(self):
        pos_proteins = []
        neg_proteins = []
        proteins = self.protein_map.keys()
        for protein in proteins:
            find_protein = False
            for true_protein in self.true_proteins:
                if protein.startswith(true_protein):
                    find_protein = True
                    break
            if find_protein is False:
                neg_proteins.append(protein)
            else:
                pos_proteins.append(protein)
        print(f"# pos proteins: {len(pos_proteins)}, # neg proteins: {len(neg_proteins)}")
        return pos_proteins, neg_proteins

    def get_contaminate_proteins(self):
        all_proteins = [protein.split(' ')[0] for protein in list(self.get_protein_sequences(self.search_fasta).keys())]
        false_proteins =  set(all_proteins) - set(self.true_proteins)
        return [protein for protein in false_proteins if not protein.startswith('DECOY')]

    def get_customized_protein_name(self):
        contaminate_proteins = self.get_contaminate_proteins()
        custom_protein_map = dict()
        for protein in self.protein_map:
            if protein in contaminate_proteins:
                custom_protein_map[protein] = "CONTAMINATE_"+protein
            else:
                custom_protein_map[protein] = protein
        return custom_protein_map









