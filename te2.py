#!/usr/bin/env python
# coding: utf-8

# In[34]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import *
import numpy as np
import sympy as sp
from scipy.spatial.distance import pdist, squareform
import math

#### Molécules #####

ethene = np.array([[-0.5, 0], [0.5, 0]])
benzene = np.array([[0.00000,1.40272], [-1.21479, 0.70136], [-1.21479, -0.70136],
                    [0.00000, -1.40272], [1.21479, -0.70136], [1.21479, 0.70136]])

### je choisis la molécule (la liste de coordonnées)

molecule2 = benzene

### FONCTIONS #### - ben du code pour le centre de masse, mais focusons sur l'hamitlonien plus bas

# Define what is considered close for a series of points
def sont_proches_(p, q, critere=1.2):
    return sum((x - y)**2 for (x, y) in zip(p, q)) <= critere**2

# Normalisation of bond length
# Assuming molecule_ is previously defined and is a numpy array of atomic positions
mindist = np.min(pdist(molecule2))
molecule = (1.00 / mindist) * molecule2

def matrice_adjacence(matrice_atomes):
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

# Mass center algebra
def replaceDiagonal(matrix, replacementList):
    for i in range(len(replacementList)):
        matrix[i][i] = replacementList[i]

def add_vectors(*points):
    new_x = sum(point[0] for point in points)
    new_y = sum(point[1] for point in points)
    return [new_x, new_y]

def subtract_vectors(a, b):
    return [a[0] - b[0], a[1] - b[1]]

def mul_by_scalar(vector, scalar):
    return [vector[0] * scalar, vector[1] * scalar]

def near(a, b, rtol=1e-5, atol=1e-8):
    return np.abs(a - b) < (atol + rtol * np.abs(b))

def imprimer(objet):
    # Assuming debug is a boolean that controls whether to print the object
    debug = False  # Or True if you want to print
    if debug:
        print(objet)
        
def reflectance_SSP(matrice_atomes, beta, E):
    ell = len(matrice_atomes)
    C= E * np.identity(ell, dtype=complex)
    matrice_contacts_abs = matrice_atomes
    matrice_contacts_abs[0][0] = -1j*beta
    matrice_contacts_abs[-1][-1] = -1j*beta
    det_abs = np.linalg.det(matrice_contacts_abs - C)
######   
    matrice_contacts_trans = matrice_atomes
    matrice_contacts_trans[0][0] = 1j*beta
    matrice_contacts_trans[-1][-1] = -1j*beta
    det_trans = np.linalg.det(matrice_contacts_trans - C)

    reflectance = (det_trans/det_abs)

    return reflectance

# generer l'axe des X ou l'énergie des électrons qui traversent la molécule
def gen_E():
    for E in np.arange(start=-3, stop= 6, step= 0.001):
        yield E

#lcp =left contact position and rcp = right contact position et j'index avec LC et RC -1


def reflectance_SSP(matrice_atomes, beta, E, lccc, rccc):
    ell = len(matrice_atomes)
    LC = lccc - 1
    RC = rccc - 1

    matrice_contacts_abs = matrice_atomes.astype(complex)
    matrice_contacts_trans = matrice_atomes.astype(complex)

    C = E * np.identity(ell, dtype=complex)
    
    matrice_contacts_abs[LC][LC] = -1j * beta
    matrice_contacts_abs[RC][RC] = -1j * beta
    
    
    det_abs = np.linalg.det(matrice_contacts_abs - C)
    
    matrice_contacts_trans[LC][LC] = 1j * beta
    matrice_contacts_trans[RC][RC] = -1j * beta
    det_trans = np.linalg.det(matrice_contacts_trans - C)

    reflectance = (det_trans / det_abs)

    return reflectance

def transmission(matrice_atome, beta):
    liste_E = gen_E()
    liste_T_E = []
    for e in liste_E:
        reflectance = reflectance_SSP(matrice_atome, beta, e, lccc, rccc)
        reflectance2 =np.absolute(reflectance) ** 2
        transmission = 1 - (reflectance2)
        liste_T_E.append([transmission, e])
        
    return liste_T_E

        
        
# Variables pour l'hamiltonien
beta_val = float(input('Beta Contact: '))  
lccc = int(input('Position du contact gauche (source) sur la molecule :'))
rccc = int(input('Position du contact droit (puit) sur la molecule :'))

# Adjust coordinates

CM = np.mean(molecule, axis=0)
new_molecule = []

for point in molecule:
    new_molecule.append(subtract_vectors(point, CM))  # Fixed order of subtraction

##Ceci genere l'hamiltonien de Huckel

hamiltonien = matrice_adjacence(new_molecule) * -1

transmission_mol = transmission(hamiltonien, beta_val)

transmission_values, energies = zip(*transmission_mol)

plt.figure(figsize=(10, 6))
plt.plot(energies, transmission_values, label='Transmission Coefficient')
plt.title('Transmission vs Energy')
plt.xlabel('Energy (arb. units)')
plt.ylabel('Transmission')
plt.grid(True)
plt.legend()
plt.show()

