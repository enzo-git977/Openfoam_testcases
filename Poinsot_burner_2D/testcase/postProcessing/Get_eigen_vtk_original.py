print("Loading libraries...")
import numpy as np 
import matplotlib.pyplot as plt
import cantera as ct
import os
import sys

# search specific libraries
current_dir = os.getcwd()  # Gets current working directory
pycsp_path = os.path.join(current_dir, 'PyCSP')  # Joins to form path to PyCSP
sys.path.insert(0, pycsp_path)  # Adds to sys.path so we can import from PyCSP

import PyCSP.Functions as csp # import Pycsp
from PyCSP.ThermoKinetics import CanteraThermoKinetics # import Pycsp

# === Paths =========================================================================================================
filename = os.path.join(current_dir, "cuttingPlanes", "0.037", "zMid.vtk") # vtk file path
mech_dir = os.path.join(current_dir, 'csp-main','mech','mechanism.yaml')  # mech file path
output_dir = 'Eigen_data'
os.makedirs(output_dir, exist_ok=True)      # dir where figures are stored

# === Mechanism =====================================================================================================
gas_foam = csp.CanteraCSP(mech_dir)
#gas_foam.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
gas_foam.constP = True
species = gas_foam.species_names + ['T']
print("species_OF = ", species)

# === Load OpenFOAM Data ============================================================================================
plane_vtk = csp.CanteraCSP.read_vtk_plane(filename, plane="xy", value=0.0005, tol=1e-6) # if mech changes change it also in the Function.py in pycsp

points = plane_vtk["points"]
T_array = plane_vtk["temperature"]
P_array = plane_vtk["pressure"]
Y_dict = plane_vtk["species"]
n_points = len(points)
species_names = gas_foam.species_names  # full mechanism species

print(f"Number of points in the plane: {points.shape[0]}")

# [<what to include> for <variable(s)> in <iterable> if <condition>]
print("Non zero species : ", [sp for sp, arr in Y_dict.items() if np.any(arr > 1e-12)] )

#======== OpenFoam eigenvalues/vectors ==============================================================================
evals = []
Revec = []
Levec = []
#fvec = []

print(f"Computing eigenvalues using {gas_foam.jacobiantype} kernel")
for i in range(n_points):
    T = T_array[i]
    P = P_array[i]
    Y = np.array([Y_dict[sp][i] if sp in Y_dict else 0.0 for sp in species_names])
    
    # skip points where reactive species are too small
    if Y.sum() < 1e-12:
        continue
    
    gas_foam.TPY = T, P, Y
    lam, R, L, _ = gas_foam.get_kernel()
    
    evals.append(lam)
    Revec.append(R)
    Levec.append(L)
    #fvec.append(f)

# convert to arrays
evals = np.array(evals)
Revec = np.array(Revec)
Levec = np.array(Levec)
#fvec = np.array(fvec)

print("Eigenvalues and eigenvectors obtained.")


# === Compute radical pointers =========================================================================================
rad_all = []
for i in range (n_points):
    rad_i = csp.CSP_pointers(Revec[i], Levec[i])
    rad_all.append(rad_i)

rad_all = np.array(rad_all)

# Max real eigenvalue indices
max_idx = np.argmax(evals.real, axis=1)

# Extract radical pointers for max eigenvalue 
rad_pointers_max = np.array([rad_all[i, idx] for i, idx in enumerate(max_idx)])
denom = np.sum(np.abs(rad_pointers_max), axis=1, keepdims=True)
denom[denom==0]=1 # avoid 0 division
rad_pointers_max = rad_pointers_max / denom  # Normalize per species (so they sum to 1 per point)
rad_pointers_max = np.abs(rad_pointers_max.real)

print("Radical pointers obtained successfully")

#======== Save Results ==============================================================================
np.savez_compressed(
     os.path.join(output_dir, 'eigen_data.npz'),
    evals=evals,
    Revec=Revec,
    Levec=Levec,
    rad_pointers_max = rad_pointers_max
)
