# CS6208_project
Code base for CS6208 project

### Raw Data:
------------
The raw data used for creating our graphs is made available in this repo.

* 9606.protein.info.v11.5.txt for gene-gene interactions
* DrugSimDB_v1_0_0.csv for drug-drug similarity
* GDSC2_fitted_dose_response_24Jul22.xlsx for cell line response to drugs
* Model.csv for metadata on cell lines, used to create cell line node features
* chemBL_drugs.csv for constructing drug node features using SMILES
* drugbank_vocabulary.csv for drug metadata
* gene_pathways.txt for constructing gene node features from the pathways each gene belongs to
* interactions.tsv for information on the gene each drug targets
* OmicsSomaticMutations.csv to get the list of genes mutated in each cell line

### Graph Construction:
----------------------
To construct the heterogeneous graph please refer to the folder "Graph Construction". The final version of the graph can be downloaded from 

### Benchmarks:
--------------
We compare our method against existing methods for (1) heterogeneous graphs - RGCN and HGT as well as (2) existing cell line drug response prediction models (GraphCDR). These can be found in the folder `benchmarks`

### Our Proposed Model:
----------------------
Our model is implemented for the cell line drug response prediction task in the notebook `our_model.ipynb`
