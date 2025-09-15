print("Loading libraries...")
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os
import sys

# search specific libraries
current_dir = os.getcwd()  # Gets current working directory
pycsp_path = os.path.join(current_dir, 'PyCSP')  # Joins to form path to PyCSP
csp_main_path = os.path.join(current_dir, 'csp-main')  # Joins to form path to csp-main
sys.path.insert(0, pycsp_path)  # Adds to sys.path so we can import from PyCSP
sys.path.insert(0, csp_main_path)  # Adds to sys.path so we can import from csp-main

import PyCSP.Functions as csp # import Pycsp
from PyCSP.ThermoKinetics import CanteraThermoKinetics # import Pycsp

from plotter import CSPPlotter  # import plotter 

# === Paths =========================================================================================================
npz_path = 'Eigen_data/eigen_data.npz'
filename = os.path.join(current_dir, "cuttingPlanes", "0.037", "zMid.vtk") # vtk file path
mech_dir = os.path.join(current_dir, 'csp-main','mech','mechanism.yaml')  # mech file path
output_dir = 'figures_vtk'
os.makedirs(output_dir, exist_ok=True)      # dir where figures are stored


# === Mechanism =====================================================================================================
gas_foam = csp.CanteraCSP(mech_dir)
#gas_foam.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
gas_foam.constP = True
species = gas_foam.species_names + ['T']
print("species_OF = ", species)

# === Load VTK Data ===================================================================================================
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

# === Load eigen Data ============================================================================================
eigen_data = np.load(npz_path)

evals = eigen_data['evals']                       # eigenvalues
Revec = eigen_data['Revec']                       # Right eigenvectors
Levec = eigen_data['Levec']                       # Left eigenvectors
rad_pointers_max = eigen_data['rad_pointers_max'] # Radical pointers

print("Eigenvalues and vectors loaded successfully")

# === Compute explosiveness ============================================================================================
max_eig_per_point = np.max(evals.real, axis=1)  # max eigenval per point
explosive_mask = max_eig_per_point > 1e-12      # Create a boolean mask for explosive points

colors = np.zeros(len(max_eig_per_point))       # Create a color array: 1 for explosive, 0 for non-explosive
colors[explosive_mask] = 1                      # explosive points

# === Compute Radical pointers =========================================================================================
rad_max_val = np.max(rad_pointers_max, axis=1)       # value of strongest radical pointer
rad_max_idx = np.argmax(rad_pointers_max, axis=1)    # index of strongest radical
rad_max_name = [species[i] for i in rad_max_idx]
for i in range(10):
        print(f"Point{i} : strongest radiacl {rad_max_name[i]} with value {rad_max_val[i]:.3e}")
        
        
# Filter only explosive points (for fig4)
explosive_idx = np.where(explosive_mask)[0]  # indices of explosive points
rad_max_name_expl = [rad_max_name[i] for i in explosive_idx]
points_expl = points[explosive_idx, :]

# === Plotting =========================================================================================================
csp_plotter = CSPPlotter(
    vtk_file=filename,  
    p=P_array,
    Y=Y_dict,
    x=points[:, 0],       
    x_indices=[0],        
    eigenvalues=max_eig_per_point,
    vl=None,
    cmap=plt.cm.inferno
)

_, _, _, _, triang, _ = csp_plotter.mesh_data

"""
# === Figure 1: Explosiveness ===
fig1, ax1 = plt.subplots(figsize=(8,6))
cmap = plt.cm.Reds 
cmap_custom = ListedColormap(['lightgray', 'red']) # Use a colormap: 0 = gray (non-explosive), 1 = red (explosive)
norm = BoundaryNorm([0, 0.5, 1], cmap_custom.N)
cf1 = ax1.tripcolor(triang, colors, cmap=cmap_custom, norm=norm, shading='flat')
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_aspect('equal')
ax1.set_title("Explosive points in the plane")
plt.savefig(f"{output_dir}/Explosiveness.png", dpi=150)
plt.close()
"""

# === Figure 2: Temperature field ===
fig2, ax2 = plt.subplots(figsize=(8,6))
cf2 = ax2.tripcolor(triang, T_array, cmap=plt.cm.inferno, shading='gouraud')
fig2.colorbar(cf2, ax=ax2, label='Temperature [K]')
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_aspect('equal')
plt.savefig(f"{output_dir}/Temperature.png", dpi=150)
plt.close()

# === Figure 3: Radical pointers magnitude ===
fig3, ax3 = plt.subplots(figsize=(8,6))

cf3 = ax3.tripcolor(triang, rad_max_val, cmap=plt.cm.viridis, shading='gouraud')
fig3.colorbar(cf3, ax=ax3, label='Radical Pointer Magnitude')
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_aspect('equal')
plt.savefig(f"{output_dir}/Radical_pointers_values.png", dpi=150)
plt.close()

# === Figure 4 : Radical pointers index (species) ===
fig4, ax4 = plt.subplots(figsize=(8,6))

# Assign a unique color to each species
unique_species = list(dict.fromkeys(rad_max_name_expl))  # unique species preserving order
species_fixed_colors = {
    "T": "orange",
    "O2": "blue",
    "H2O": "deepskyblue",
    "CO2": "green",
    "CO": "red",
    "C3H8": "purple",
    "N2": "black"
}
species_to_color = {}
for sp in unique_species:
    if sp in species_fixed_colors:
        species_to_color[sp] = species_fixed_colors[sp]
        
ax4.scatter(points[:,0], points[:,1], s=10, color='lightgray', alpha=0.3)
for sp in unique_species:
    idx = [i for i, name in enumerate(rad_max_name_expl) if name == sp]
    ax4.scatter(points_expl[idx,0], points_expl[idx,1], s=10, color=species_to_color[sp], label=sp)

ax4.set_xlabel("x [m]")
ax4.set_ylabel("y [m]")
ax4.set_aspect('equal')

# Legend horizontally below
ax4.legend( 
    title= 'Species',
    loc='upper center',       # centered above or below the axes
    bbox_to_anchor=(0.5, -0.80),  # below the plot
    ncol=len(unique_species),      # one column per species
    frameon=False,
    fontsize=12
)
plt.savefig(f"{output_dir}/Radical_dominant_species.png", dpi=150)
plt.show()
#plt.close()

# === End ===
print(f"plotting successful, result saved in {output_dir}")

