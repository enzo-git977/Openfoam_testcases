print("Loading libraries...")
import numpy as np 
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d

# import Pycsp
current_dir = os.getcwd()  # Gets current working directory
pycsp_path = os.path.join(current_dir, 'PyCSP')  # Joins to form path to PyCSP
sys.path.insert(0, pycsp_path)  # Adds to sys.path so we can import from PyCSP

import PyCSP.Functions as csp
from PyCSP.ThermoKinetics import CanteraThermoKinetics

# === Paths =======================================================================================================
current_dir = os.getcwd()  # Gets current working directory
Cantera_path = 'Cantera_solution_Stoich/data_new_mech/cantera_profile.csv'
time = "0.01"
OF_path = os.path.join(current_dir, "Openfoam_solution_Stoich", "postProcessing", "axialLine", time, "line1.xy")
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)                   # dir where figures are stored
os.makedirs('data_eigen_Stoichiometric/', exist_ok=True) # dir where eigenvalues and vectors data are stored

# === Load Cantera Data ==========================================================================================
with open(Cantera_path) as f:
    header = f.readline().lstrip('#').strip().split(',')

df_cantera = pd.read_csv(Cantera_path, comment='#', header=None)
df_cantera.columns = header

x_cantera = df_cantera['Position_mm'].values
T_cantera = df_cantera['Temperature_K'].values
species_cantera = {
    'H': df_cantera['H'].values,
    'O': df_cantera['O'].values,
    'H2': df_cantera['H2'].values,
    'O2': df_cantera['O2'].values,
    'OH': df_cantera['OH'].values,
    'H2O': df_cantera['H2O'].values,
    'N2': df_cantera['N2'].values,
    'N': df_cantera['N'].values,
    'NO': df_cantera['NO'].values,
    'HO2': df_cantera['HO2'].values,
    'H2O2': df_cantera['H2O2'].values,
    'AR': df_cantera['AR'].values,
    'OH*': df_cantera['OH*'].values,
    'HE': df_cantera['HE'].values,
}


print("Loaded Cantera data")

# === Load OpenFOAM Data =========================================================================================
foam_data = pd.read_csv(OF_path, comment='#', header=None,sep='\s+')
foam_data.columns = ['x','T','H','O','H2','O2','OH','H2O','N2','N','NO','HO2','H2O2','AR','OH*','HE']

x_foam = foam_data['x'].values * 1000  # convert m to mm to match Cantera
T_foam = foam_data['T'].values
species_foam = {
     'H': foam_data['H'].values,
    'O': foam_data['O'].values,
    'H2': foam_data['H2'].values,
    'O2': foam_data['O2'].values,
    'OH': foam_data['OH'].values,
    'H2O': foam_data['H2O'].values,
    'N2': foam_data['N2'].values,
    'N': foam_data['N'].values,
    'NO': foam_data['NO'].values,
    'HO2': foam_data['HO2'].values,
    'H2O2': foam_data['H2O2'].values,
    'AR': foam_data['AR'].values,
    'OH*': foam_data['OH*'].values,
    'HE': foam_data['HE'].values,
}

print("Loaded OpenFOAM data")

print("size ct",x_cantera.shape)
print("size of",x_foam.shape)


#======== Cantera eigenvalues/vectors ==============================================================================
# Mechanism
gas_cantera = csp.CanteraCSP('mech/New_h_mech.yaml')
phi = 1 # stoichiometric
gas_cantera.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
pressure = ct.one_atm
T_in = 300.0
gas_cantera.constP = True
gas_cantera.TP = T_in, pressure
species = gas_cantera.species_names + ['T']
print("species_Cantera = ", species)

# Initialize containers
evals_cantera = []
Revec_cantera = []
Levec_cantera = []
fvec_cantera = []
# get eigenvalues and eigenvectors for every timestep
for i in range(len(x_cantera)):
    T_1 = T_cantera[i]
    Y_1 = np.zeros(gas_cantera.n_species)
    # Fill species mass fractions
    for j, sp in enumerate(gas_cantera.species_names):
        if sp in df_cantera.columns:
            Y_1[j] = df_cantera[sp][i]
    # Set Cantera gas state to update kernel
    gas_cantera.TPY = T_1, pressure, Y_1
    lam_cantera, R_cantera, L_cantera, f_cantera = gas_cantera.get_kernel()
    evals_cantera.append(lam_cantera)
    Revec_cantera.append(R_cantera)
    Levec_cantera.append(L_cantera)
    fvec_cantera.append(f_cantera)
# convert to array
evals_cantera = np.array(evals_cantera)
Revec_cantera = np.array(Revec_cantera)
Levec_cantera = np.array(Levec_cantera)
fvec_cantera = np.array(fvec_cantera)
print("Eigenvalues and eigenvectors for Cantera simulation obtained")  


#======== OpenFoam eigenvalues/vectors ==============================================================================
# Mechanism
gas_foam = csp.CanteraCSP('mech/New_h_mech.yaml')
gas_foam.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
gas_foam.constP = True
gas_foam.TP = T_in, pressure
species = gas_foam.species_names + ['T']
print("species_OF = ", species)

# Initialize containers
evals_foam = []
Revec_foam = []
Levec_foam = []
fvec_foam = []

# Iterate over spatial points (x_foam)
for i in range(len(x_foam)):
    T = T_foam[i]
    Y = np.zeros(gas_foam.n_species)
    
    # Fill species mass fractions
    for j, sp in enumerate(gas_foam.species_names):
        if sp in foam_data.columns:
            Y[j] = foam_data[sp].iloc[i]
    
    # Set state and get kernel
    gas_foam.TPY = T, pressure, Y
    lam_foam, R_foam, L_foam, f_foam = gas_foam.get_kernel()
    
    evals_foam.append(lam_foam)
    Revec_foam.append(R_foam)
    Levec_foam.append(L_foam)
    fvec_foam.append(f_foam)

# Convert results to arrays
evals_foam = np.array(evals_foam)
Revec_foam = np.array(Revec_foam)
Levec_foam = np.array(Levec_foam)
fvec_foam = np.array(fvec_foam)

print("Eigenvalues and eigenvectors for Foam simulation obtained.")

# === Shifting so that both plot can be compared ==================================================================
# Compute the temperature peak positions
T_target = 1500
idx_peak_cantera = np.argmin(np.abs(T_cantera - T_target)) # Cantera
idx_peak_foam = np.argmin(np.abs(T_foam - T_target)) # OF
x_peak_cantera = x_cantera[idx_peak_cantera]
x_peak_foam = x_foam[idx_peak_foam]

# Compute the shift to apply to OpenFOAM to align with Cantera
shift = x_peak_cantera - x_peak_foam
print(f" Aligning peaks: shift OpenFOAM by {shift:.3f} mm")

# Apply shift
x_foam_shifted = x_foam + shift


# === Save eignevalues/vectors arrays =================================================================================
np.savez_compressed(
    'data_eigen_Stoichiometric/eigen_data.npz',
    evals_foam=evals_foam,
    Revec_foam=Revec_foam,
    Levec_foam=Levec_foam,
    evals_cantera=evals_cantera,
    Revec_cantera=Revec_cantera,
    Levec_cantera=Levec_cantera
)
print("Eigenvalues and vectors saved to data_eigen_Stoichiometric/eigen_data.npz")


# === Plot explosiveness =================================================================================
explosive_cantera = np.max(evals_cantera.real, axis=1) > 1e-7
explosive_foam = np.max(evals_foam.real, axis=1) > 1e-7

# === Plot temperature with explosive regions highlighted ===
plt.figure(figsize=(10, 6))
plt.plot(x_cantera, T_cantera, label='Cantera', color='blue')
plt.plot(x_foam_shifted, T_foam, label='OpenFOAM', color='green')

# Mark explosive regions with red dots or shading
plt.scatter(x_cantera[explosive_cantera], T_cantera[explosive_cantera], color='red', s=10, label='Explosive (Cantera)')
plt.scatter(x_foam_shifted[explosive_foam], T_foam[explosive_foam], color='orange', s=10, label='Explosive (OpenFOAM)')

plt.xlabel('x [mm]')
plt.ylabel('Temperature [K]')
plt.title('Temperature Profile with Explosive Regions')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(os.path.join(output_dir, 'explosiveness_temperature_plot.png'), dpi=300)
print(f"plotting successful, result saved in {output_dir}/explosiveness_temperature_plot.png")
plt.show()
