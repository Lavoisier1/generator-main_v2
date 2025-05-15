import sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import pandas as pd

def is_planar(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_hs = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol_hs, useRandomCoords=True, maxAttempts=1000)
    try:
        AllChem.MMFFOptimizeMolecule(mol_hs)

        try:
            pbf = rdMolDescriptors.CalcPBF(mol_hs)
        except Exception:
            return False

        # Check if the molecule is planar and print the result
        if pbf < 1.:
            print(f"Molecule {smiles} is planar. PBF={pbf}")
            return True
        else:
            print(f"Molecule {smiles} is NOT planar. PBF={pbf}")
            return False
    except Exception:
        return False


# Path to your CSV file
csv_path = sys.argv[1]
# Path to the output CSV file
output_csv_path = sys.argv[2]

# Number of molecules to calculate
num_mols = None
if len(sys.argv) == 4:
    num_mols = int(sys.argv[3])

# Read the CSV file
df = pd.read_csv(csv_path)
if num_mols:
    df = df.head(num_mols).copy()

# Apply the function and filter the DataFrame
filtered_df = df[df['smiles'].apply(is_planar)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_csv_path, index=False)

