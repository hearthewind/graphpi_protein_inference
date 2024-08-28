# GraphPI Protein Inference


## Folder Structure
    └── code: contains the source code for the project.
        └── main_train.py: main file to train the model.
        └── main_eval.py: main file to make predictions.
        └── configs.py : contains the configuration for the project.
    └── data: contains our test data, which includes:
        └── PSM features.
        └── Fasta database (these two constitute the input to our algorithm). 
        └── Results from other algorithms.
    └── results: our trained model is saved here, also includes:
        └── Protein groups.
        └── Prediction made by our algorithm.
    └── knime_workflows: contains the knime workflows.
        └── generate_decoy: generate decoy database based on fasta file.
        └── comet_search: execute peptide search algorithm
        └── epifany_pipeline: run epifany on Percolator results, also includes Percolator to generate PSM features.
        └── fido_pipeline, pia_pipeline: run fido and pia.

## Usage

### Environment Setup

For environment preparing, please use conda:
conda env create -f environment.yml

### Train

#### Pretrain data

We use data from the public repository (promeXchange) for training purpose, 
the link to download the raw files are included in Table S2 in supporting information.
Please run the following procedure to acquire training features.

1. Download the raw files from proteomeXchange, the fasta database from Uniprot.
2. Run the generate_decoy knime workflow to generate a decoy fasta database.
3. Run the comet_search knime workflow for peptide database search and Percolator.
4. Run the epifany_pipeline knime workflow to generate the psm and epifany scores.
5. Collect the psm and epifany scores for all training data.

To run this program for demonstration, we provide the psm and epifany scores for one single dataset which is stored in data/PXD005388/.

#### Run
Run main_train.py to train the model.

### Inference

#### Test data

We use what data... and their raw files can be download from...
and we stored their psm features json files in ...

#### Run
You can run the following commands to train the model


