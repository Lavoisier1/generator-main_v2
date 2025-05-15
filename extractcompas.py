import csv
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#### Fonctions #####

def _homo_lumo_gap_liste(liste):
    valpropre_ordonee = sorted(liste)
    print (valpropre_ordonee)
    longeur_eigenval = len(valpropre_ordonee)
    if (longeur_eigenval == 2):
        homo_pos = longeur_eigenval - 1
        lumo_pos = longeur_eigenval
        homo = valpropre_ordonee[homo_pos]
        lumo = valpropre_ordonee[lumo_pos]
        diff = homo - lumo
        details = [homo, lumo]
        return details
    if (longeur_eigenval % 2 == 0):
        homo_pos = (longeur_eigenval//2) - 1
        lumo_pos = (longeur_eigenval//2)
        homo = valpropre_ordonee[homo_pos]
        lumo = valpropre_ordonee[lumo_pos]
        diff = homo - lumo
        details = [homo, lumo]
        return details
    else:
        homo_pos = int(longeur_eigenval//2) - 1
        lumo_pos = int(longeur_eigenval//2)
        homo = valpropre_ordonee[homo_pos]
        lumo = valpropre_ordonee[lumo_pos]
        details = [homo, lumo]
        return details

# define a function to generate adjacency matrix for a given molecule
def get_adjacency_matrix(mol):
    """
    Generate the adjacency matrix for a given molecule.
    """
    size = mol.GetNumAtoms()
    adj = [[0] * size for i in range(size)]
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        adj[begin_atom][end_atom] = 1
        adj[end_atom][begin_atom] = 1
    return adj


####### Importer toute la DB  #####

with open('COMPAS-1D.csv', newline='') as csvfile:
    # Create a reader object
    reader = csv.reader(csvfile)
    print(reader)

    # Extract the desired column and append its elements to a new list
    smiles = []
    for row in reader:
        smiles.append(row[1])
    smiles.pop(0)

# generate adjacency matrix for each molecule
adj_list = []
for s in smiles:
    mol = Chem.MolFromSmiles(s)
    adj = get_adjacency_matrix(mol)
    adj_list.append((s, adj))

# multiply by -1 to have \beta -1
adj_list_neg = []
for i in range(len(adj_list)):
    mat = adj_list[i][1]
    s = adj_list[i][0]
    adj_list_neg.append((s, np.multiply(mat, -1)))

# print all adjacency matrices
#for i, adj in enumerate(adj_list_neg):
#    print(f"Molecule {i+1}:")
#    for row in adj:
#        print(row)
#    print("\n")
#transformer la liste de matrice d'ajacence en eigensysten

eigen_list = []

for e in adj_list_neg:
    matrix = e[1]
    s = e[0]
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigen_list.append((s, matrix, eigvals, eigvecs))

new_eigen_list = []

for s, matrix, eigvals, eigvecs in eigen_list:
    #determiner la position du eigenvectors de HOMO
    longeur_eigenval = len(eigvals)
    homo_pos = (longeur_eigenval//2) - 1
    lumo_pos = (longeur_eigenval//2)
    line_indices = [homo_pos, lumo_pos]
    print("line indices", line_indices)
    add_to_list = False
    for idx in line_indices:
        if any(elem <= 1.0E-10 for elem in eigvecs[idx]):
            add_to_list = True
            break
    if add_to_list:
        new_eigen_list.append((s, matrix, eigvals, eigvecs))

print(len(new_eigen_list))
for i, adj in enumerate(new_eigen_list):
    print(f"Molecule {i+1}:")
    for row in adj:
        print(row)
    exit(0)
    print("\n")

print (len(new_eigen_list))
