import os
import tempfile
import argparse
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.spatial.distance import pdist, squareform

from plotting import plot_graph_with_highlighted_contact, plot_transmission, plot_mol_contacts, combine_all_images

class Graph:
    """
    Graph data structure for representing molecules.

    Attributes:
        V (int): The number of vertices in the graph.
        graph (defaultdict): A dictionary that represents an adjacency list of the graph.
    """
    def __init__(self, vertices):
        """
        Initialize a new Graph.

        Args:
            vertices (int): The number of vertices in the graph.
        """
        self.V = vertices
        self.graph = {i: [] for i in range(vertices)}

    def add_edge(self, v, w):
        """
        Add an edge between two vertices in the graph.

        Args:
            v (int): The starting vertex of the edge.
            w (int): The ending vertex of the edge.
        """
        self.graph[v].append(w)
        self.graph[w].append(v)

    def carbon_with_three_bonds(self, v):
        """
        Determine if a carbon vertex is connected to exactly three other carbons.

        Args:
            v (int): The vertex to check.

        Returns:
            bool: True if the vertex is a carbon with three bonds, False otherwise.
        """
        # Count the number of carbon neighbors
        carbon_neighbors = sum(1 for neighbor in self.graph[v] if neighbor in self.graph)
        return carbon_neighbors == 3

    def unique_pairs_from_vertex(self, start):
        """
        Find all unique pairs starting from a given vertex.

        Args:
            start (int): The vertex to start the search from.

        Returns:
            list of tuples: A list of unique pairs found from the start vertex.
        """
        visited = [False] * self.V
        queue = [start]
        pairs = []

        while queue:
            vertex = queue.pop(0)
            for i in self.graph[vertex]:
                if not visited[i]:
                    queue.append(i)
                    # Exclude the starting vertex and carbons with 3 carbon bonds
                    if i != start and not self.carbon_with_three_bonds(i):
                        pairs.append((start, i))
            visited[vertex] = True

        return pairs

    def unique_pairs(self):
        """
        Find all unique pairs in the graph.

        Returns:
            list of tuples: A list of all unique pairs in the graph.
        """
        all_pairs = []
        for v in range(self.V):
            if not self.carbon_with_three_bonds(v): # Check here to skip carbons with 3 carbon bonds
                all_pairs.extend(self.unique_pairs_from_vertex(v))
        
        # Remove duplicate pairs
        all_pairs = list(set([tuple(sorted(pair)) for pair in all_pairs]))

        return all_pairs

def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if not molecule:
        raise ValueError("Invalid SMILES string")
    
    # Ensure molecule has been kekulized to deal with aromaticity correctly
    Chem.Kekulize(molecule)
    
    g = Graph(molecule.GetNumAtoms())

    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(start, end)
    
    return g

def generate_contacts(smiles_string, plot=False):
    """
    Generate contact pairs from a SMILES string and optionally plot them.

    Args:
        smiles_string (str): The SMILES string of the molecule.
        plot (bool, optional): Flag to determine whether to plot the contacts.

    Returns:
        list of tuples: A sorted list of unique contact pairs.
    """
    g = smiles_to_graph(smiles_string)
    sorted_contacts = sorted(g.unique_pairs(), key=lambda x: x[0])
    if plot:
        plot_mol_contacts(smiles_string, sorted_contacts)

    return sorted_contacts

#def generate_adjacency_matrix(smiles_string):
#    """
#    Generate the adjacency matrix for a given SMILES string of a molecule.
#
#    Args:
#        smiles_string (str): The SMILES string of the molecule.
#
#    Returns:
#        list of lists: The adjacency matrix of the molecule.
#    """
#    mol = Chem.MolFromSmiles(smiles_string)
#    size = mol.GetNumAtoms()
#    adj = [[0] * size for i in range(size)]
#    for bond in mol.GetBonds():
#        begin_atom = bond.GetBeginAtomIdx()
#        end_atom = bond.GetEndAtomIdx()
#        adj[begin_atom][end_atom] = 1
#        adj[end_atom][begin_atom] = 1
#
#    adj = np.multiply(adj, -1)
#    return adj

def sont_proches_(p, q, critere=1.2):
    """
    Define what is considered close for a series of points
    """
    return sum((x - y) ** 2 for (x, y) in zip(p, q)) <= critere**2


def generate_adjacency_matrix(smiles):
    """
    Generate the adjacency matrix for a given SMILES string of a molecule.

    Args:
        smiles_string (str): The SMILES string of the molecule.

    Returns:
        list of lists: The adjacency matrix of the molecule.
    """

    # Create a molecule object from the SMILES string found in COMPAS DB
    molecule = Chem.MolFromSmiles(smiles)

    # Compute 2D coordinates for the molecule
    Chem.rdDepictor.Compute2DCoords(molecule)

    # Get the 2D coordinates
    atoms = molecule.GetAtoms()
    coords = [molecule.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in atoms]
    x_coords = [coord.x for coord in coords]
    y_coords = [coord.y for coord in coords]

    # Transformer le SMILES EN X, Y, Z et ensuite en matrice d'Adjacence

    combined_coords = np.array(list(zip(x_coords, y_coords)))

    # Normalisation de la longuer de la liason
    mindist = np.min(pdist(combined_coords))
    matrice_atomes = (1.00 / mindist) * combined_coords

    ell = len(matrice_atomes)
    matrice = np.zeros((ell, ell))
    for i in range(0, ell):
        atome_i = matrice_atomes[i]
        for j in range(i + 1, ell):
            atome_j = matrice_atomes[j]
            if sont_proches_(atome_i, atome_j):
                matrice[i, j] = 1
                matrice[j, i] = 1
    return matrice

    return adjacency_matrix

def check_evecs(row):
    """
    Analyze the eigenvalues and eigenvectors of a row from a DataFrame.

    Args:
        row (pd.Series): A row of DataFrame containing the adjacency matrix and its eigen values and vectors.

    Returns:
        bool: True if specific condition is met, False otherwise.
    """
    matrix, eigvals, eigvecs = row["adj_matrix"], row["eigvals"], row["eigvecs"]

    # determiner la position du eigenvectors de HOMO
    longeur_eigenval = len(eigvals)
    homo_pos = (longeur_eigenval//2) - 1
    lumo_pos = (longeur_eigenval//2)
    line_indices = [homo_pos, lumo_pos]
    add_to_list = False

    for idx in line_indices:
        if any(elem <= 1.e-10 for elem in eigvecs[idx]):
            return True

def print_results(row):
    """
    Print the results contained within a DataFrame row.

    Args:
        row (pd.Series): A row from the DataFrame.
    """
    print(f"Molecule\n {row['smiles']} {row['adj_matrix']} {row['eigvals']}")

def generate_energy_range(min_energy, max_energy, step=0.001):
    """
    Generate the electrons' energies that traverse the molecules.
    """
    for E in np.arange(start=min_energy, stop=max_energy, step=step):
        yield E

def calculate_reflectance(matrice_atomes, beta, E, lccc, rccc):
    """
    Calculate the reflectance of a molecule (eq. 18)
    """
    ell = len(matrice_atomes)
    LC = lccc - 1
    RC = rccc - 1
    matrice_contacts_abs = matrice_atomes.astype(complex)
    matrice_contacts_trans = matrice_atomes.astype(complex)
    C = E * np.identity(ell, dtype=complex)

    # numerator of eq. 18 or Absorbant Hamiltonien
    matrice_contacts_abs[LC][LC] = -1j * beta
    matrice_contacts_abs[RC][RC] = -1j * beta
    det_abs = np.linalg.det(matrice_contacts_abs - C)

    # denominator of eq. 18 or Transparent Hamiltonien
    matrice_contacts_trans[LC][LC] = 1j * beta
    matrice_contacts_trans[RC][RC] = -1j * beta
    det_trans = np.linalg.det(matrice_contacts_trans - C)

    # full eq. 18
#    print(f"|Ht| = {det_trans}")
#    print(f"|Ha| = {det_abs}")
    if np.abs(det_trans) < 1.0e-8 and np.abs(det_abs) < 1.0e-8:
        reflectance = 0
    else:
        reflectance = (det_trans / det_abs)

#    print(f"r = {reflectance}")
#    print()

    return reflectance

def calculate_transmission(matrice_atome, beta, lccc, rccc, min_energy, max_energy, auto_adjust=False):
    """
    Calculate the transmision probability of a molecule: T(E) = 1 - r^{2}(E)
    """
    if auto_adjust:
        if min_energy < 0:
            min_energy -= 2
        else:
            min_energy += 2
        if max_energy < 0:
            max_energy -= 2
        else:
            max_energy += 2

    energies = list(generate_energy_range(min_energy=min_energy, max_energy=max_energy, step=0.05))

    transmissions = []
    for e in energies:
        reflectance = calculate_reflectance(matrice_atome, beta, e, lccc, rccc)
        transmission = 1. - np.absolute(reflectance) ** 2
        transmissions.append(transmission)
    return energies, transmissions

def worker(args):
    adj_matrix, contact_pair = args
    lccc, rccc = contact_pair
    energies, transmissions = calculate_transmission(adj_matrix, 0.1, lccc, rccc, min_energy=-3, max_energy=3, auto_adjust=False)
    return energies, transmissions

def calculate_row_transmission(row, max_workers):
    """
    Calculate the transmission using a pandas row.

    Extract the adjacency_matrix and the contacts from the row that represents a molecule, add
    artificial contacts and generate the unique contacts for the target molecule.

    Parameters:
    row (pandas row): A pandas row that contains a molecule and its properties.
    Returns
    -------
    unique_contacts (list): A list containing the unique contacts.
    transmissions (list): A list containing the transmissions of each contact.
    """
    adj_matrix, contacts = row["adj_matrix"], row["contacts"]
    mat_adj_contacts, mat2pair = add_artifical_contacts(adj_matrix, contacts)
    unique_mol, unique_contacts = unique_molecules_to_matrices(mat_adj_contacts, mat2pair)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [(adj_matrix, contact_pair) for contact_pair in unique_contacts]
        futures = list(executor.map(worker, args))
    
    # futures = [(energies, transmissions), ...]
    # Collect results after the executor context
    results = []
    for future in futures:
        try:
            results.append(future)
        except Exception as e:
            print(f"Error occurred: {e}")

    # results is a list of tuples of 2 lists (energies, transmissions)
    return unique_contacts, results

def add_artifical_contacts(matrix, liste):
    """
    Add a methyl and an ethyl to the first and second contacts, respectively.
    """
    modified_matrices = []
    matrix_to_pair = {}  # Initialize the dictionary here

    for pair in liste:
        # Create a new matrix with extra space for three new atoms (one for the connection and two for the ethyl group)
        size = matrix.shape[0] + 3
        new_matrix = np.zeros((size, size))

        # Copy the original matrix into the new matrix
        new_matrix[0 : matrix.shape[0], 0 : matrix.shape[0]] = matrix

        # Add the new connections
        atom1, atom2 = pair
        # Connect the first atom to the first new atom (part of the ethyl group)
        new_matrix[atom1, matrix.shape[0]] = 1
        new_matrix[matrix.shape[0], atom1] = 1

        # Connect the second atom to the second new atom (part of the ethyl group)
        new_matrix[atom2, matrix.shape[0] + 1] = 1
        new_matrix[matrix.shape[0] + 1, atom2] = 1

        # Connect the two atoms of the ethyl group
        new_matrix[matrix.shape[0], matrix.shape[0] + 2] = 1
        new_matrix[matrix.shape[0] + 2, matrix.shape[0]] = 1

        # Add the new matrix to the list and map it to the pair
        modified_matrices.append(new_matrix)
        matrix_to_pair[
            tuple(new_matrix.flatten())
        ] = pair  # Use the flattened matrix as a key

    return modified_matrices, matrix_to_pair

def matrix_to_molecule(adj_matrix):
    mol = Chem.RWMol()  # Create an empty molecule

    # Add atoms
    for i in range(adj_matrix.shape[0]):
        mol.AddAtom(Chem.Atom(6))  # 6 is the atomic number for Carbon

    # Add bonds
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if (
                adj_matrix[i, j] == 1
            ):  # Adjust this condition based on your matrix values
                mol.AddBond(i, j, Chem.BondType.SINGLE)  # Adjust bond type as needed

    return mol

def unique_molecules_to_matrices(list_of_matrices, matrix_to_pair):
    unique_smiles = set()
    unique_matrices = []
    unique_pairs = []

    for matrix in list_of_matrices:
        mol = matrix_to_molecule(matrix)
        smiles = Chem.MolToSmiles(mol, canonical=True)

        if smiles not in unique_smiles:
            unique_smiles.add(smiles)
            unique_matrices.append(matrix)
            unique_pairs.append(
                matrix_to_pair[tuple(matrix.flatten())]
            )  # Retrieve the pair using the flattened matrix

    return unique_matrices, unique_pairs

def main():

    parser = argparse.ArgumentParser(description="Run the Reorgs")
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--molecule", type=str, help="Smiles string of target molecule")
    group.add_argument("-d", "--dataset", type=str, help="Dataset file")

    parser.add_argument("-n", "--number", default=-1, type=int, help="Number of molecules. Default: process all molecules")
    parser.add_argument("-p", "--plt", default=False, type=bool, help="Plot contacts")
    args = parser.parse_args()

    if args.dataset:
        # Read dataset into DataFrame and workout a dataset
        df = pd.read_csv(args.dataset)
        # Apply the generate_contacts function to the 'smiles' column
        if args.number == -1:
            df['contacts'] = df['smiles'].apply(generate_contacts)
        elif args.number > -1:
            df = df.head(args.number).copy()
            df['contacts'] = df['smiles'].apply(lambda x: generate_contacts(x, args.plt))

        print(f"Number of rows in df: {len(df.index)}")
        df['adj_matrix'] = df['smiles'].apply(generate_adjacency_matrix)
        df[['eigvals', 'eigvecs']] = df['adj_matrix'].apply(lambda x: pd.Series(np.linalg.eigh(x)))

        result_series = df.apply(lambda row: calculate_row_transmission(row=row, max_workers=36), axis=1)

        # result_series = [(energies, transmissions), ...]
        contacts = []
        transmissions = []
        for result in result_series:
            contacts_row = result[0]
            contacts.append(contacts_row)
            transmissions_row = result[1]
            transmissions.append(transmissions_row)

        contacts = [result[0] for result in result_series]
        transmissions = [result[1] for result in result_series]

        """
        contacts = [
                    [(contact 1, contact 2), (contact 3, contact 4), ...], #  contacts of molecule 1
                      .
                      .
                      .
                    [(contact 1, contact 2), (contact 3, contact 4), ...], #  contacts of molecule n
        ]

        transmissions = [
                            [(energies, transmissions), ...], # energies and transmissions of contacts 1 and 2 of molecule 1
                            .
                            .
                            .
                            [(energies, transmissions), ...], # energies and transmissions of contacts 1 and 2 of molecule n
        ]

        Note that energies and transmissions are lists.
        """

        df['contacts'] = contacts
        df['transmissions'] = transmissions

        df.to_csv('DATASETS/MODIFIED_COMPAS-1D.csv', index=False)

    elif args.molecule:
        # Work out a single molecule
        contacts = generate_contacts(args.molecule, args.plt)
        adj_matrix = generate_adjacency_matrix(args.molecule)
        eigvals, eigvecs = np.linalg.eigh(adj_matrix)
        min_energy = np.min(eigvals)
        max_energy = np.max(eigvals)
        print(f"Eigen values: \n{eigvals}")
        mat_adj_contacts, mat2pair = add_artifical_contacts(adj_matrix, contacts)
        unique_mol, unique_pairs = unique_molecules_to_matrices(mat_adj_contacts, mat2pair)

        all_images = []
        for lccc, rccc in unique_pairs:
            rdkit_mol = Chem.MolFromSmiles(args.molecule)
            Chem.Kekulize(rdkit_mol)
            energies, transmissions = calculate_transmission(adj_matrix, 0.1, lccc, rccc, min_energy, max_energy)
            combined_img = plot_transmission(rdkit_mol, energies, transmissions, args.molecule, lccc, rccc)
            all_images.append(combined_img)

        # Combine all images into one
        final_image = combine_all_images(all_images)
        final_image.show() 
        final_image.save("combined_plots.png") 

if __name__ == "__main__":
    main()
