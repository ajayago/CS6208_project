### Create hetero graph

import pandas as pd
import torch as th
import pickle as pkl
import numpy as np
from tqdm import tqdm
import dgl
import torch as th
from dgl.data.utils import save_graphs

# Build dict for nodes

used_node_idx = 0
node2idx_cellLine = {}

cell_line_nodes_df = pd.read_csv('/home/salman/GNNproj/node_features/cell_line_node_features.csv')
for cl in set(cell_line_nodes_df['ModelID'].tolist()):
    # print(cl); break
    node2idx_cellLine[cl] = used_node_idx
    used_node_idx += 1

print("Num of unique cell lines: {} | Node indices: {} to {}\n".format(used_node_idx, 0, used_node_idx-1))


used_node_idx = 0
node2idx_drug = {}

drug_node_df = pd.read_csv('/home/salman/GNNproj/node_features/drug_node_features.csv')
for dname in set(drug_node_df['Name'].tolist()):
    # print(cl); break
    node2idx_drug[dname] = used_node_idx
    used_node_idx += 1

print("Num of unique drugs: {} | Node indices: {} to {}\n".format(used_node_idx, 0, used_node_idx-1))


used_node_idx = 0
node2idx_gene = {}

gene_node_df = pd.read_csv('/home/salman/GNNproj/node_features/gene_node_features.csv')
for gname in set(gene_node_df['genes'].tolist()):
    # print(cl); break
    node2idx_gene[gname] = used_node_idx
    used_node_idx += 1

print("Num of unique genes: {} | Node indices: {} to {}\n".format(used_node_idx, 0, used_node_idx-1))


def get_drug_drug_edges():
    df = pd.read_csv('/home/salman/GNNproj/link_features/drug_drug_similarity.csv')

    # NEW
    # Column Names: ['Unnamed: 0', 'Chem_similarity', 'Target_similarity', 
    #    'Pathway_similarity', 'GO_CC_Similarity', 'GO_MF_Similarity', 
    #    'GO_BP_Similarity', 'DRUG_NAME', 'DRUG_NAME2'] 
    feat_column_list = ['Chem_similarity', 'Target_similarity', 'GO_CC_Similarity', 'GO_MF_Similarity', 'GO_BP_Similarity']
    num_feats = len(feat_column_list)
    
    print(len(df))
    # print(df.head(3))

    sources = []
    destinations = []
    # NEW
    edge_feat_mat = []

    for key, row in tqdm(df.iterrows()):
        # print(key, row) ; break
        drug1 = row['DRUG_NAME']
        drug2 = row['DRUG_NAME2']
        sources.append(node2idx_drug[drug1])
        destinations.append(node2idx_drug[drug2])
        # print(sources, destinations)

        # NEW
        edge_feat = row[feat_column_list].values.astype(np.float64)
        edge_feat = th.from_numpy(edge_feat)
        edge_feat_mat.append(th.unsqueeze(edge_feat, dim=0)) # Unsqueeze to form row matrix. Concat along axis=0 at the end.

    sources = th.tensor(sources)
    destinations = th.tensor(destinations)

    # NEW
    edge_feat_tensor = th.cat(edge_feat_mat, axis = 0)
    return (sources, destinations, edge_feat_tensor)

# x = get_drug_drug_edges()
# print(x[0].shape, x[1].shape, x[2].shape)


def get_drug_gene_edges():
    """
    NO EDGE FEAT, RETURN PAIR NOT TRIPLET
    """
    df = pd.read_csv('/home/salman/GNNproj/link_features/gene_drug_target.csv')
    print(len(df))
    # print(df.head(3))
    sources = []
    destinations = []
    for key, row in tqdm(df.iterrows()):
        # print(key, row) ; break
        drug = row['DRUG_NAME']
        gene = row['HugoSymbol']
        sources.append(node2idx_drug[drug])
        destinations.append(node2idx_gene[gene])
        # print(sources, destinations)
        # break
    sources = th.tensor(sources)
    destinations = th.tensor(destinations)
    return (sources, destinations)

# len(get_drug_gene_edges()[0])


def get_cl_drug_edges():
    df = pd.read_csv('/home/salman/GNNproj/link_features/cl_drug_response.csv')

    # NEW
    # Column Names: ['Unnamed: 0', 'DRUG_NAME', 'AUC', 'DepMap_ID']
    feat_column_list = ['AUC']
    num_feats = len(feat_column_list)

    print(len(df))
    # print(df.head(3))
    sources = []
    destinations = []
    
    # NEW
    edge_feat_mat = []

    for key, row in tqdm(df.iterrows()):
        # print(key, row) ; break
        cl = row['DepMap_ID']
        drug = row['DRUG_NAME']
        sources.append(node2idx_cellLine[cl])
        destinations.append(node2idx_drug[drug])
        # print(sources, destinations)
        # NEW
        edge_feat = row[feat_column_list].values.astype(np.float64)
        edge_feat = th.from_numpy(edge_feat)
        edge_feat_mat.append(th.unsqueeze(edge_feat, dim=0)) # Unsqueeze to form row matrix. Concat along axis=0 at the end.

    sources = th.tensor(sources)
    destinations = th.tensor(destinations)

    # NEW
    edge_feat_tensor = th.cat(edge_feat_mat, axis = 0)
    return (sources, destinations, edge_feat_tensor)

# x = get_cl_drug_edges()
# print(x[0].shape, x[1].shape, x[2].shape)


def get_gene_gene_edges(from_file=True):
    """
    node2idx depends on csv files, tolist() and set() func.
    Remember to RERUN with from_file=False upon changes in files (or system?).
    """
    if from_file:
        with open('/home/salman/GNNproj/link_features/Gene_gene_edges_tensor_pair.pkl', 'rb') as f:
            sources, destinations, edge_feat_tensor = pkl.load(f)
    else:
        df = pd.read_csv('/home/salman/GNNproj/link_features/gene_gene_interaction.csv')
        
        # NEW
        # Column Names: ['Unnamed: 0', 'combined_score', 'HugoSymbol', 'HugoSymbol2']
        feat_column_list = ['combined_score']
        num_feats = len(feat_column_list)
        
        print(len(df))
        # print(df.head(3))
        sources = []
        destinations = []

        # NEW
        edge_feat_mat = []

        for key, row in tqdm(df.iterrows()):
            # print(key, row) ; break
            gene1 = row['HugoSymbol']
            gene2 = row['HugoSymbol2']
            sources.append(node2idx_gene[gene1])
            destinations.append(node2idx_gene[gene2])
            # print(sources, destinations)
            
            # NEW
            edge_feat = row[feat_column_list].values.astype(np.float64)
            edge_feat = th.from_numpy(edge_feat)
            edge_feat_mat.append(th.unsqueeze(edge_feat, dim=0)) # Unsqueeze to form row matrix. Concat along axis=0 at the end.

        sources = th.tensor(sources)
        destinations = th.tensor(destinations)

        # NEW
        edge_feat_tensor = th.cat(edge_feat_mat, axis = 0)
        
        with open('/home/salman/GNNproj/link_features/Gene_gene_edges_tensor_pair.pkl', 'wb') as f:
            pkl.dump((sources, destinations, edge_feat_tensor), f)

    return (sources, destinations, edge_feat_tensor)

# x = get_gene_gene_edges(from_file=False)
# print(x[0].shape, x[1].shape, x[2].shape)


def get_cl_gene_edges(from_file=True):
    """
    node2idx depends on csv files, tolist() and set() func.
    Remember to RERUN with from_file=False upon changes in files (or system?).
    """
    if from_file:
        with open('/home/salman/GNNproj/link_features/CellLine_gene_edges_tensor_pair.pkl', 'rb') as f:
            sources, destinations = pkl.load(f)
    else:
        df = pd.read_csv('/home/salman/GNNproj/link_features/cl_gene_mutations.csv')
        print(len(df))
        # print(df.head(3))
        sources = []
        destinations = []
        for key, row in tqdm(df.iterrows()):
            # print(key, row) ; break
            cl = row['DepMap_ID']
            gene = row['HugoSymbol']
            sources.append(node2idx_cellLine[cl])
            destinations.append(node2idx_gene[gene])
            # print(sources, destinations)
            # break
        sources = th.tensor(sources)
        destinations = th.tensor(destinations)
        with open('/home/salman/GNNproj/link_features/CellLine_gene_edges_tensor_pair.pkl', 'wb') as f:
            pkl.dump((sources, destinations), f)
    return (sources, destinations)

# len(get_cl_gene_edges()[0])

# Define the heterograph (only connections)

# Create a heterograph with 3 node types and 3 edges types.

# Triplets (src, dest, feat): get_drug_drug_edges, get_cl_drug_edges, get_gene_gene_edges
# Pairs (src, dest): get_drug_gene_edges, get_cl_gene_edges
s1, d1, edge_feat_drug_drug = get_drug_drug_edges()
s2, d2                      = get_drug_gene_edges()
s3, d3, edge_feat_cl_drug   = get_cl_drug_edges()
s4, d4, edge_feat_gene_gene = get_gene_gene_edges(from_file=False)
s5, d5                      = get_cl_gene_edges(from_file=False)

graph_data = {
   ('drug', 'interacts', 'drug'): (s1,d1),
   ('drug', 'interacts', 'gene'): (s2,d2),
   ('cell_line', 'response', 'drug'): (s3,d3),
   ('gene', 'interacts', 'gene'): (s4,d4),
   ('cell_line', 'mutates', 'gene'): (s5,d5)
   ### ADD reverse edges later
}
g = dgl.heterograph(graph_data)

print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)

def get_node_attrib_tensor(node_type, attrib_csv):
    """
    node_type: edge type to deal with
    attrib_csv: corresponding attribute file

    Use node2idx for finding index
    Read csv and assign attributes iteratively
    """
    if node_type == 'gene':
        attrib_df = pd.read_csv(attrib_csv) #0-th column is the gene name in this csv
    else:
        attrib_df = pd.read_csv(attrib_csv, index_col=0)
    # print(attrib_df.head(3))

    if node_type == 'cell_line':
        """
        attributes are 42 binary columns.
        There are 1710 cellLines. -> len(node2idx_cellLine)
        """
        num_features = 42 # len(attrib_df.columns)-1
        num_nodes = len(node2idx_cellLine) # 1710
        feat_tensor = th.zeros(num_nodes, num_features, dtype=th.float64)

        column_list = attrib_df.columns.tolist()
        feat_column_list = column_list[:-1]
        # print(feat_column_list)

        for idx, row in tqdm(attrib_df.iterrows()):
            # print(idx)
            # print(row)
            node_index = node2idx_cellLine[row['ModelID']]
            # print(node_index)
            node_feat = row[feat_column_list].values.astype(np.float64)
            node_feat = th.from_numpy(node_feat)
            # print(node_feat)
            feat_tensor[node_index, :] = node_feat
            # print(feat_tensor[node_index])
            # break
    
    elif node_type == 'drug':
        """
        Attributes are fingerprints 2048 bits (fp_0 to fp_2047), Name, Smiles
        There are 1889 drugs. -> len(node2idx_drug)
        """
        num_features = 2048 # len(attrib_df.columns)-2, exclude Name and Smiles
        num_nodes = len(node2idx_drug) # 1889
        feat_tensor = th.zeros(num_nodes, num_features, dtype=th.float64)

        column_list = attrib_df.columns.tolist()
        feat_column_list = column_list[:-2]
        # print(feat_column_list)

        for idx, row in tqdm(attrib_df.iterrows()):
            # print(idx)
            # print(row)
            node_index = node2idx_drug[row['Name']]
            # print(node_index)
            node_feat = row[feat_column_list].values.astype(np.float64)
            node_feat = th.from_numpy(node_feat)
            # print(node_feat)
            feat_tensor[node_index, :] = node_feat
            # print(feat_tensor[node_index])
            # break

    elif node_type == 'gene':
        """
        Attributes are:  genes, 3090 binary columns
        There are 27236 genes. -> len(node2idx_gene)
        """
        num_features = 3090 # len(attrib_df.columns)-1, exclude Name and Smiles
        num_nodes = len(node2idx_gene) # 27236
        feat_tensor = th.zeros(num_nodes, num_features, dtype=th.float64)

        column_list = attrib_df.columns.tolist()
        feat_column_list = column_list[1:] # Skip the first column which is gene name
        # print(feat_column_list)

        for idx, row in tqdm(attrib_df.iterrows()):
            # print(idx)
            # print(row)
            node_index = node2idx_gene[row['genes']]
            # print(node_index)
            node_feat = row[feat_column_list].values.astype(np.float64)
            node_feat = th.from_numpy(node_feat)
            # print(node_feat)
            feat_tensor[node_index, :] = node_feat
            # print(feat_tensor[node_index])
            # break    
    
    else:
        feat_tensor = th.zeros(1,1)
        print("Invalid node type!!!")

    return feat_tensor


# cellLine_feat_tensor = get_node_attrib_tensor('cell_line', '/home/salman/GNNproj/node_features/cell_line_node_features.csv')
# drug_feat_tensor = get_node_attrib_tensor('drug', '/home/salman/GNNproj/node_features/drug_node_features_fingerprints.csv')
# gene_feat_tensor = get_node_attrib_tensor('gene', '/home/salman/GNNproj/node_features/gene_node_features.csv')

# print(gene_feat_tensor.shape)

g.nodes['cell_line'].data['hv'] = get_node_attrib_tensor('cell_line', '/home/salman/GNNproj/node_features/cell_line_node_features.csv')
g.nodes['drug'].data['hv'] = get_node_attrib_tensor('drug', '/home/salman/GNNproj/node_features/drug_node_features_fingerprints.csv')
g.nodes['gene'].data['hv'] = get_node_attrib_tensor('gene', '/home/salman/GNNproj/node_features/gene_node_features.csv')

g.edges[('drug', 'interacts', 'drug')].data['he']  = edge_feat_drug_drug
# g.edges[('drug', 'interacts', 'gene')].data['he']  = 
g.edges[('cell_line', 'response', 'drug')].data['he']  = edge_feat_cl_drug
g.edges[('gene', 'interacts', 'gene')].data['he']  = edge_feat_gene_gene
# g.edges[('cell_line', 'mutates', 'gene')].data['he']  = 

# g.edata['train_mask'] = 

save_graphs("/home/salman/GNNproj/saved_graph.bin", g)


