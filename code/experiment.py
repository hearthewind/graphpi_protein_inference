import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from configs import PROJECT_ROOT_DIR, PROJECT_DATA_DIR
from datasets.util import get_proteins_by_fdr
from train.util import get_dataset, iprg_converter
import pickle
import os
import seaborn as sns
from datasets.mix18 import Mix18

def get_fdr_vs_TP_graphs(save_dir="default", result_dict=None, image_post_fix="default"):
    X = []
    # baselines = ["pia", "deeppep", "epifany", "fido", "mymodel"]
    # datasets = ["iPRG2016_A", "iPRG2016_B", "iPRG2016_AB", "yeast", "ups2"]
    # group_name = ["iprg_a", "iprg_b", "iprg_ab", "yeast", "ups2"]

    baselines = ["mymodel", "epifany", "fido", "pia", 'deeppep']
    # baselines = ["proteinprophet"]

    # datasets = ["iPRG2016_A", "iPRG2016_B", "iPRG2016_AB", "ups2", 'yeast', '18mix']
    # group_name = ["iprg_a", "iprg_b", "iprg_ab", 'ups2', 'yeast', '18mix']

    datasets = ['hela_3t3']
    group_name = ['hela_3t3']

    # datasets = ["iPRG2016_A", "iPRG2016_B", "iPRG2016_AB", "ups2", '18mix']
    # group_name = ["iprg_a", "iprg_b", "iprg_ab", 'ups2', '18mix']

    # datasets = ['18mix', 'ups2']
    # group_name = ['18mix', 'ups2']

    indistinguishable_groups = {}
    for i, data in enumerate(group_name):
        with open(os.path.join(PROJECT_ROOT_DIR, "outputs", f"{data}_groups.pkl"), "rb") as f:
            indistinguishable_groups[datasets[i]] = pickle.load(f)

    fdr_rates = np.arange(0, 0.15, 0.01)
    for dataset in datasets:
        data = get_dataset(dataset=dataset, process=False)
        for pretrain_data_name in baselines:
            if pretrain_data_name in ["epifany", "fido"]:
                if "iPRG2016" in dataset:
                    dataset_prefix, dataset_type = dataset.split("_")
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                              f"result_{dataset_type.lower()}.json")
                elif 'hela_3t3' in dataset:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", f"{data.data_code}.json")
                else:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", "result.json")
                protein_scores = data.extract_protein_score(protein_score_path)
            elif pretrain_data_name == "mymodel":
                if result_dict is None:
                    if "iPRG2016" in dataset:
                        dataset_prefix, dataset_type = dataset.split("_")
                        protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                                  f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                                  f"result_{dataset_type}.pkl")
                    elif 'hela_3t3' in dataset:
                        protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                                  f"{dataset}/{pretrain_data_name}_result", f"{data.data_code}.pkl")
                    else:
                        protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                                  f"{dataset}/{pretrain_data_name}_result", "result.pkl")
                    with open(protein_score_path, "rb") as f:
                        protein_scores = pickle.load(f)
                else:
                    protein_scores = result_dict[dataset]
            elif pretrain_data_name == "mymodel_td":
                if result_dict is None:
                    if "iPRG2016" in dataset:
                        dataset_prefix, dataset_type = dataset.split("_")
                        protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                                  f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                                  f"result_{dataset_type}.pkl")
                    elif 'hela_3t3' in dataset:
                        protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                                  f"{dataset}/{pretrain_data_name}_result", f"{data.data_code}.pkl")
                    else:
                        protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                                  f"{dataset}/{pretrain_data_name}_result", "result.pkl")
                    with open(protein_score_path, "rb") as f:
                        protein_scores = pickle.load(f)
                else:
                    protein_scores = result_dict[dataset]
            elif pretrain_data_name == "deeppep":
                if "iPRG2016" in dataset:
                    dataset_prefix, dataset_type = dataset.split("_")
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                              f"result_{dataset_type}.csv")
                elif 'hela_3t3' in dataset:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", f"{data.data_code}.csv")
                else:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", "result.csv")
                protein_scores = dict(pd.read_csv(protein_score_path).values)
            elif pretrain_data_name == "pia":
                if "iPRG2016" in dataset:
                    dataset_prefix, dataset_type = dataset.split("_")
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                              f"result_{dataset_type.lower()}.json")
                elif 'hela_3t3' in dataset:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", f"{data.data_code}.json")
                else:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", "result.json")
                if '18mix' in dataset:
                    protein_scores = extract_pia_18mix(protein_score_path, data)
                elif 'hela_3t3' in dataset:
                    protein_scores = extract_pia_18mix(protein_score_path, data)
                else:
                    protein_scores = extract_protein_score(protein_score_path, method="pia")
            elif pretrain_data_name == "proteinprophet":
                if "iPRG2016" in dataset:
                    dataset_prefix, dataset_type = dataset.split("_")
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                              f"result_{dataset_type.lower()}.json")
                else:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", "result.json")
                protein_scores = extract_protein_score(protein_score_path, method="protein_prophet")
            elif pretrain_data_name == "percolator":
                if "iPRG2016" in dataset:
                    dataset_prefix, dataset_type = dataset.split("_")
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset_prefix.lower()}/{pretrain_data_name}_result",
                                              f"result_{dataset_type}.json")
                else:
                    protein_score_path = join(PROJECT_ROOT_DIR, PROJECT_DATA_DIR,
                                              f"{dataset}/{pretrain_data_name}_result", "result.json")
                protein_scores = extract_protein_score(protein_score_path, method="percolator")
            for fdr_rate in fdr_rates:
                protein_scores = {iprg_converter(protein, data):score for protein, score in protein_scores.items()}
                indishtinguishable_group = {
                    iprg_converter(protein, data): [iprg_converter(protein_, data) \
                                                                       for protein_ in protein_pairs] for
                    protein, protein_pairs in indistinguishable_groups[dataset].items()}
                contaminate_proteins = [iprg_converter(protein, data) for protein in data.get_contaminate_proteins()]
                num_pos_proteins = len(
                    get_proteins_by_fdr(protein_scores, contaminate_proteins=contaminate_proteins,
                                        fdr=fdr_rate, indishtinguishable_group=indishtinguishable_group))
                X.append({"method": pretrain_data_name, "dataset": dataset, "fdr": fdr_rate, "# positive proteins": num_pos_proteins})

    X = pd.DataFrame(X)
    pd.DataFrame(X).to_csv(os.path.join(PROJECT_ROOT_DIR, "outputs", save_dir, f"results_comparison_{image_post_fix}.csv"), index=False)
    
    
    data_ = pd.DataFrame(X)
    fig, axes = plt.subplots(len(datasets),1, figsize=(10, 15))
    for i, dataset in enumerate(datasets):
        if len(datasets) == 1:
            ax = axes
        else:
            ax = axes[i]
        sns.lineplot(data=data_.query(f"dataset=='{dataset}'"), x="fdr", y="# positive proteins", hue="method", ci=None, ax=ax)
        ax.set_title(dataset)
        ax.legend(loc="right")
    plt.tight_layout()
    #plt.savefig(os.path.join(PROJECT_ROOT_DIR, "outputs", "experiment_plot.png"))
    plt.savefig(os.path.join(PROJECT_ROOT_DIR, "outputs", save_dir, f"experiment_plot_{image_post_fix}.png"))
    # plt.show()

import json
def extract_protein_score(protein_score_path, method="pia"):
    """
    Extract the protein scores from a benchmark algorithm (e.g. FIDO)
    :return:
    """
    file_name = protein_score_path
    with open(file_name, "r") as f:
        json_data = json.load(f)

    protein_scores = {}
    if method == "pia":
        for data in json_data:
            proteins = data["Accessions"]
            score = data["Score"]

            for protein in proteins:
                protein_scores[protein] = score
    elif method == "percolator":
        for data in json_data:
            proteins = data["accession"]
            score = data["best_search_engine_score[1]"]

            for protein in proteins:
                protein_scores[protein] = 1-score
    elif method == "protein_prophet":
        for result in json_data['data']:
            name = result['proteins']
            prob = result['probability']

            names = name.split(',')

            for name in names:
                protein_scores[name] = prob

    return protein_scores

def extract_pia_18mix(protein_score_path: str, mix18: Mix18):
    protein_list = [protein.split(' ')[0] for protein in list(mix18.get_protein_sequences(mix18.search_fasta).keys())]
    protein_list = [protein for protein in protein_list if not protein.startswith('DECOY')]

    protein_name_dict = {}
    for name in protein_list:
        if name.endswith('|'):
            simp_name = name.split('|')[-2]
        elif name.startswith('sp|'):
            simp_name = name.split('|')[1]
        else:
            simp_name = name

        protein_name_dict[simp_name] = name

    file_name = protein_score_path
    with open(file_name, "r") as f:
        json_data = json.load(f)

    protein_scores = {}
    for data in json_data:
        proteins = data["Accessions"]
        score = data["Score"]

        for protein in proteins:
            protein_scores[protein] = score

    ret = {}
    for simp_name, score in protein_scores.items():
        if simp_name.startswith('DECOY'):
            name = simp_name
        else:
            name = protein_name_dict[simp_name]
        ret[name] = score

    return ret


if __name__ == "__main__":
    get_fdr_vs_TP_graphs()