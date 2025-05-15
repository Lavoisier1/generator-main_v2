import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


#  genere des liste  E
def gen_E():
    for E in np.arange(start=-3, stop=3, step=0.01):
        yield E


# _ham_ = matrice_adjacence(molecule_)*-1


### Rodrigo : Ici de facto la source est sur le premier atome de la matrice d'adjacence et le puit sur le dernier atome, faut le looper avec ton code*
def reflectance_SSP(matrice_atomes, beta, E):
    """
    Calculate the determinant of the absorbant and transparent Hamiltonians.
    """
    ell = len(matrice_atomes)
    C = E * np.identity(ell, dtype=complex)
    matrice_contacts_abs = matrice_atomes
    matrice_contacts_abs[0][0] = -1j * beta
    matrice_contacts_abs[-1][-1] = -1j * beta
    # numerator of eq. 18
    det_abs = np.linalg.det(matrice_contacts_abs - C)

    matrice_contacts_trans = matrice_atomes
    matrice_contacts_trans[0][0] = 1j * beta
    matrice_contacts_trans[-1][-1] = -1j * beta
    # denominator of eq. 18
    det_trans = np.linalg.det(matrice_contacts_trans - C)

    # full eq. 18
    reflectance = det_trans / det_abs

    return reflectance


# Looper la fonction de la reflectance ci dessus


def transmission(matrice_atome, beta):
    """
    Calculate the transmision probability of a molecule
    """
    # generate range of energies
    liste_E = gen_E()

    liste_T_E = []
    for e in liste_E:
        reflectance = reflectance_SSP(matrice_atome, beta, e)
        reflectance2 = np.absolute(reflectance) ** 2
        transmission = 1 - (reflectance2)
        liste_T_E.append([transmission, e])
    return liste_T_E


#### fonctions ####


# Une fonction pour la matrice d'adjacence
def get_adjacency_matrix(mol):
    size = mol.GetNumAtoms()
    adj = [[0] * size for i in range(size)]
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        adj[begin_atom][end_atom] = 1
        adj[end_atom][begin_atom] = 1
    return adj


####### Importer toute la DB  #####

dataset = sys.argv[1]
with open(dataset, newline="") as csvfile:
    # Create a reader object
    reader = csv.reader(csvfile)
    print(reader)

    # Extract the desired column and append its elements to a new list
    smiles = []
    for row in reader:
        smiles.append(row[1])
    smiles.pop(0)

# generate adjacency matrix & geometric distance matrix & T(E) for each molecule, la distance geometrique c'est dans l'idée de dévelloper un nouveau descripteur moléculaire
# c'est une idée de Matthias que je trouve compliquer quand le GOA ou GOR existe (graph of rigns and graph of atoms)

adj_list = []

for s in smiles:
    mol = Chem.MolFromSmiles(s)
    adj = get_adjacency_matrix(mol)

    #### Calcul de T(E) à zero

    # Create a molecule object
    molecule = Chem.MolFromSmiles(s)

    # Add hydrogens to the molecule
    molecule = Chem.AddHs(molecule)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(molecule)

    # Remove hydrogens if not needed
    molecule = Chem.RemoveHs(molecule)

    # Get the number of carbon atoms
    num_carbons = sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "C")

    # Initialize the distance matrix
    distance_matrix = np.zeros((num_carbons, num_carbons))

    # Loop over atoms to fill the distance matrix
    carbon_index = 0
    carbon_positions = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "C":
            position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            carbon_positions.append(np.array([position.x, position.y, position.z]))

    for i in range(num_carbons):
        for j in range(i + 1, num_carbons):
            distance = np.linalg.norm(carbon_positions[i] - carbon_positions[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    # Define negative alpha
    alpha = -50

    # Calculate the sum of exponentials of the modified upper triangle elements for a range of x values
    x_values = np.linspace(1, 10, 1000)  # Adjust these values as needed
    sum_exponentials_values = []
    for x in x_values:
        upper_triangle = np.triu(distance_matrix, k=1)
        sum_exponentials = np.sum(
            (np.exp(alpha * (upper_triangle - x) ** 2))
            / (np.sqrt(2 * np.pi) / np.sqrt(-alpha))
        )
        sum_exponentials_values.append(sum_exponentials)

    print()
    #    print(sum_exponentials_values)
    exit(0)

#### Calcul de T(E) via python et linéaire fit ######
