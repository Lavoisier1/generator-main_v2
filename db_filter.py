import sys
import os
import pandas as pd
import numpy as np


def delta_z_from_xyz(path):
    """Return the z range of the molecule after PCA alignment.

    Parameters
    ----------
    path : str
        Path to the ``.xyz`` file describing the molecule.

    Returns
    -------
    float
        Difference between maximum and minimum z coordinate after
        aligning the molecule's principal plane to ``xy``.
    """

    with open(path) as f:
        lines = f.read().strip().splitlines()

    if len(lines) < 3:
        raise ValueError(f"{path} does not look like a valid xyz file")

    coords = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=float)

    # Center the coordinates
    coords -= coords.mean(axis=0)

    # PCA to obtain the principal axes
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    normal = vh[2]

    z_coords = coords.dot(normal)
    return z_coords.max() - z_coords.min()


def is_planar(molecule_name, xyz_dir, threshold):
    """Return ``True`` if the molecule is planar within ``threshold``."""

    xyz_path = os.path.join(xyz_dir, f"{molecule_name}.xyz")
    if not os.path.isfile(xyz_path):
        print(f"XYZ file not found for {molecule_name}: {xyz_path}")
        return False

    try:
        dz = delta_z_from_xyz(xyz_path)
    except Exception as exc:
        print(f"Failed to analyse {xyz_path}: {exc}")
        return False

    print(f"Molecule {molecule_name} delta_z={dz:.3f}")
    return dz <= threshold

def main():
    """Filter molecules of ``csv_path`` based on planarity."""

    if len(sys.argv) < 5:
        print(
            "Usage: python db_filter.py <input.csv> <output.csv> <xyz_dir> <delta_z_threshold> [num]"
        )
        sys.exit(1)

    csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]
    xyz_dir = sys.argv[3]
    threshold = float(sys.argv[4])

    num_mols = int(sys.argv[5]) if len(sys.argv) > 5 else None

    df = pd.read_csv(csv_path)
    if num_mols:
        df = df.head(num_mols).copy()

    filtered_df = df[df["molecule"].apply(lambda m: is_planar(m, xyz_dir, threshold))]
    filtered_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()

