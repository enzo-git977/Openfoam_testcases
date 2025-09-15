print("Importing libraries")
# === Libraries ======================================================================================================
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


# search specific libraries
current_dir = os.getcwd()                        # Gets current working directory
pycsp_path = os.path.join(current_dir, 'PyCSP')  # Joins to form path to PyCSP
sys.path.insert(0, pycsp_path)                   # Adds to sys.path so we can import from PyCSP

import PyCSP.Functions as csp                          # import Pycsp Functions
from PyCSP.ThermoKinetics import CanteraThermoKinetics # import Pycsp Thermokinectics

# === Paths ======================================================================================================
#filename = os.path.join(current_dir, "getEulerianFields", "0.037", "airCloud.dat") # filename path
filename = sys.argv[1]     # e.g., getEulerianFields/0.037/airCloud.dat
time_name = sys.argv[2]    # e.g., "0.037"
mech_dir = os.path.join(current_dir, 'csp-main','mech','mechanism.yaml')  # mech file path
output_dir = os.path.join("fig_parcel", time_name)                        # output dir  where figures are stored
os.makedirs(output_dir, exist_ok=True)                                    

# === Parcel =====================================================================================================
# Choose target parcel  
target_parcel_id = 110
print(f"parcel tracked : {target_parcel_id}")

# === Mechanism ==================================================================================================
gas = csp.CanteraCSP(mech_dir)
#gas_foam.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
gas.constP = True
species = gas.species_names + ['T']
print("species = ", species)


# === Extract data from parcel ===================================================================================

# === CREATE DATAFRAME ===
data = pd.read_csv(filename, sep=r'\s+', comment='#', header=None)

# === EXCTRACT HEADER ===
with open(filename, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line.lstrip('#').strip().split()
            break

data.columns = header  

# === EXCTRACT Parcel row ===  
parcel_rows = data[data['PARCEL_ID'] == target_parcel_id] # we want the info of the target_parcel_id
if parcel_rows.empty:
    print(f"No data found for parcel {target_parcel_id}. Available parcels: {data['PARCEL_ID'].unique()}")
    exit(1)

# === EXCTRACT TEMPERATURE AND PRESSURE of a given parcel === 
T = parcel_rows.iloc[0]['T']
p = parcel_rows.iloc[0]['p']

# === Extract species mass fractions ===
Y_list = []
for sp in gas.species_names:
    if sp in parcel_rows.columns:
        #print("Y =", sp)
        Y_list.append(parcel_rows.iloc[0][sp])
    else:
        Y_list.append(0.0)

Y = np.array(Y_list)

#======== Eigenvalues/vectors ==============================================================================
gas.TPY = T, p, Y
lam, R, L, _ = gas.get_kernel()

print("Eigenvalues and eigenvectors obtained.")

#======== Radical pointers ==============================================================================
rad = csp.CSP_pointers(R, L)

# Extract radical pointers for max eigenvalue
max_idx = np.argmax(lam.real)                     # Largest mode index
rad_pointers_max = rad[max_idx]                   # Pick that mode
rad_pointers_max = np.abs(rad_pointers_max.real)  # Absolute values
rad_pointers_max /= rad_pointers_max.sum()        # Normalization

print("Radical pointers obtained.")

#======== API  ==========================================================================================
ctk  = CanteraThermoKinetics(mech_dir)                                        # create instance
ctk.TPY = T, p, Y                                                             # set thermochemical state in ctk 
Smat = ctk.generalized_Stoich_matrix_const_p()                                # get Stoich matrix
Rates = ctk.Rates_vector()                                                    # get Reaction rates vector
API = csp.CSP_amplitude_participation_indices(L, Smat, Rates, normtype='abs') # abs : normalize so that sum of absolute values per species is 1

# ---- Identify dominant radical ----
largest_idx = np.argmax(rad_pointers_max)
dominant_species = species[largest_idx]
print(f"Largest radical pointer: {dominant_species} = {rad_pointers_max[largest_idx]:.4f}")

# API for dominant radical
api_for_species = API[largest_idx, :].real  # shape: [number_of_reactions]

# ---- Build labels for fwd/bwd ----
n_dirs = api_for_species.size
y = np.arange(n_dirs)
base_eqns = ctk.reaction_equations()         # only 2 (without direction)
labels = []
for eqn in base_eqns:
    labels.append(eqn + " (f)")
    labels.append(eqn + " (b)")



#======== Plotting ======================================================================================
# === Figure 1: Radical pointers ===
fig1, ax1 = plt.subplots(figsize=(12,6))

bars = plt.bar(species, rad_pointers_max, color='navy')

# Highlight the species with largest pointer
largest_idx = np.argmax(rad_pointers_max)
bars[largest_idx].set_color('firebrick')

ax1.set_xlabel('Species')
ax1.set_ylabel('Radical Pointer Value')
ax1.set_title(f'Parcel {target_parcel_id} Radical Pointers')
plt.setp(ax1.get_xticklabels(), ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.savefig(f"{output_dir}/Radical_pointers_values.png", dpi=150)
#plt.close()
plt.show()


# === Plot top reactions for dominant radical ===
fig2, ax2 = plt.subplots(figsize=(12,6))

plt.barh(y, api_for_species, color='navy')
ax2.set_yticks(y)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Participation Index (API)')
ax2.set_title(f'Reactions for Radical: {dominant_species}')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




