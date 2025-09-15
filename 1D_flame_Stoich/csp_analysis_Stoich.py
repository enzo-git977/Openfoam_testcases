print("Loading libraries...")
import numpy as np 
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.patches import Patch

current_dir = os.getcwd()  # Gets current working directory
pycsp_path = os.path.join(current_dir, 'PyCSP')  # Joins to form path to PyCSP
sys.path.insert(0, pycsp_path)  # Adds to sys.path so we can import from PyCSP

import PyCSP.Functions as csp
from PyCSP.ThermoKinetics import CanteraThermoKinetics

# Latex format
plt.rc('text',usetex='True')
plt.rc('font',family='serif',size=21)

# === Paths ===============================================================================================================================
npz_path = 'data_eigen_Stoichiometric/eigen_data.npz'
Cantera_path = 'Cantera_solution_Stoich/data_new_mech/cantera_profile.csv'
time = "0.01"
OF_path = os.path.join(current_dir, "Openfoam_solution_Stoich", "postProcessing", "axialLine", time, "line1.xy")
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)                   # dir where figures are stored



# === Mechanism ===================================================================================================================
gas = csp.CanteraCSP('mech/New_h_mech.yaml')
phi = 1 # stoichiometric
gas.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
pressure = ct.one_atm
T_in = 300.0
gas.constP = True
gas.TP = T_in, pressure
species = gas.species_names + ['T']
print("species = ", species)

# === Load Cantera Data ===================================================================================================================
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


print("Loaded Cantera thermochemical data")

# === Load OpenFOAM Data ==================================================================================================================
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

print("Loaded OpenFOAM thermochemical data")


# === Load eigenvalues and vectors ========================================================================================================
eigen_data = np.load(npz_path)
# Cantera
evals_cantera = eigen_data['evals_cantera']
Revec_cantera = eigen_data['Revec_cantera']
Levec_cantera = eigen_data['Levec_cantera']
print("Eigenvalues and vectors loaded successfully for Cantera")
# Openfoam
evals_foam = eigen_data['evals_foam']
Revec_foam = eigen_data['Revec_foam']
Levec_foam = eigen_data['Levec_foam']
print("Eigenvalues and vectors loaded successfully for Openfoam")

# === Shifting so that both plot can be compared ==========================================================================================
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

# === Radical pointers for several points for Cantera =====================================================================================
target_T1 = 1015
target_T2 = 1609
target_T3 = 1851.26
target_T4 = 2379.47

idx_T1 = np.argmin(np.abs(T_cantera - target_T1))
idx_T2 = np.argmin(np.abs(T_cantera - target_T2))
idx_T3 = np.argmin(np.abs(T_cantera - target_T3)) 
idx_T4 = np.argmin(np.abs(T_cantera - target_T4))
idx = [idx_T1, idx_T2, idx_T3, idx_T4]
#idx = sorted(idx)
n_points = len(idx)

# Extract corresponding CSP data from Cantera
evals_points_ct = evals_cantera[idx]        # (n_points, n_species)
Revec_points_ct = Revec_cantera[idx]        # (n_points, n_species, n_species)
Levec_points_ct = Levec_cantera[idx]        # (n_points, n_species, n_species)

# Compute radical pointers for Cantera
rad_all_ct = []
for i in range(n_points):
    rad_i = csp.CSP_pointers(Revec_points_ct[i], Levec_points_ct[i])
    rad_all_ct.append(rad_i)

rad_all_ct = np.array(rad_all_ct)

# Max real eigenvalue indices
max_idx_ct = np.argmax(evals_points_ct.real, axis=1)

# Extract radical pointers for max eigenvalue (Cantera)
rad_pointers_max_ct = np.array([rad_all_ct[i, idx] for i, idx in enumerate(max_idx_ct)])
rad_pointers_max_ct = rad_pointers_max_ct / np.sum(np.abs(rad_pointers_max_ct), axis=1, keepdims=True) # Normalize per species (so they sum to 1 per point)
rad_pointers_max_ct = np.abs(rad_pointers_max_ct.real)

print("Radical pointers for Cantera computed successfully")


# === Radical pointers for several points for Openfoam ====================================================================================
idx_T1_foam = np.argmin(np.abs(T_foam -target_T1))
idx_T2_foam = np.argmin(np.abs(T_foam - target_T2))
idx_T3_foam = np.argmin(np.abs(T_foam - target_T3))
idx_T4_foam = np.argmin(np.abs(T_foam - target_T4))
idx_foam = [idx_T1_foam, idx_T2_foam, idx_T3_foam, idx_T4_foam]
idx_foam = sorted(idx_foam)
n_points = len(idx_foam)

# Extract corresponding CSP data from Openfoam
evals_points_of = evals_foam[idx_foam]        # (n_points, n_species)
Revec_points_of = Revec_foam[idx_foam]        # (n_points, n_species, n_species)
Levec_points_of = Levec_foam[idx_foam]        # (n_points, n_species, n_species)

# Compute radical pointers for Openfoam
rad_all_of = []
for i in range (n_points):
    rad_i = csp.CSP_pointers(Revec_points_of[i], Levec_points_of[i])
    rad_all_of.append(rad_i)

rad_all_of = np.array(rad_all_of)

# Max real eigenvalue indices
max_idx_of = np.argmax(evals_points_of.real, axis=1)

# Extract radical pointers for max eigenvalue (Openfoam)
rad_pointers_max_of = np.array([rad_all_of[i, idx] for i, idx in enumerate(max_idx_of)])
rad_pointers_max_of = rad_pointers_max_of / np.sum(np.abs(rad_pointers_max_of), axis=1, keepdims=True) # Normalize per species (so they sum to 1 per point)
rad_pointers_max_of = np.abs(rad_pointers_max_of.real)

print("Radical pointers for OF obtained successfully")

print("Values for the 4 points")
for i in range(n_points):
    print(f"Cantera: {x_cantera[idx[i]]:.3f} mm ; {T_cantera[idx[i]]:.1f} K")
    print(f"OF : {x_foam_shifted[idx_foam[i]]:.3f} mm ; {T_foam[idx_foam[i]]:.1f} K")


# === Combined Radical Pointers Plot for Cantera and OpenFOAM =============================================================================
x = np.arange(len(species))  # bar positions for species
width = 0.3                  # narrower bars for side-by-side comparison

fig, axs = plt.subplots(2, 2, figsize=(18, 8), sharey=True)

for i in range(n_points):
    ax = axs.flat[i]
    # Offset bars for comparison
    ax.bar(x - width/2, rad_pointers_max_ct[i], width, label='Cantera', color='navy')
    ax.bar(x + width/2, rad_pointers_max_of[i], width, label='OpenFOAM', color='firebrick')

    ax.set_xticks(x)
    ax.set_xticklabels(species, ha='right', fontsize=8)
    ax.set_ylabel("Radical Pointer Value" if i % 3 == 0 else "")
    ax.set_title(
    rf"$x \approx {x_cantera[idx[i]]:.1f}$ mm, $T \approx {T_cantera[idx[i]]:.0f}$ K"
)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    if i == 0:
        ax.legend(fontsize=12, frameon=False)

#plt.suptitle("Radical Pointers (Max Eigenvalue) for selected points \nComparison: Cantera vs OpenFOAM", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.ylim(0,1)
plt.savefig(f"{output_dir}/radical_pointer_Cantera_vs_OpenFOAM.pdf", dpi=300)
print(f"plotting successful, result saved in {output_dir}/radical_pointer_Cantera_vs_OpenFOAM.pdf")


#===== Participation indexes at given position ============================================================================================
# Get id of target_T1
idx_point = 1 # from 0 to 3 since they are 4 pts
idx_of = idx_foam[idx_point]# OpenFOAM
idx_ct = idx[idx_point] # cantera

# Optional: Print exact times being compared
print(f"Cantera Temperature: {T_cantera[idx_ct]:.1f} K")
print(f"Openfoam Temperature:  {T_foam[idx_of]:.1f} K")

idx_max_point_ct = np.argmax(rad_pointers_max_ct[idx_point]) # cantera
idx_max_point_of = np.argmax(rad_pointers_max_of[idx_point]) # of

############   CANTERA   ####################################################
print("size Levec ct: ",Levec_points_ct.shape)
B = Levec_points_ct[idx_point,:,:] # API for given time
ctk = CanteraThermoKinetics("mech/New_h_mech.yaml")

# set thermochemical state in ctk 
T_sel = T_cantera[idx_ct]  # from Cantera
Y_sel = df_cantera.loc[df_cantera.index[idx_ct], gas.species_names].values
ctk.TPY = T_sel, pressure, Y_sel

# Computes Stoich matrix, reaction rates vector and then API
Smat = ctk.generalized_Stoich_matrix_const_p() # get Stoich matrix
Rates = ctk.Rates_vector() # get Reaction rates vector
API = csp.CSP_amplitude_participation_indices(B, Smat, Rates, normtype='abs') # abs : normalize so that sum of absolute values per species is 1

print(f"Largest radical pointer species index: {idx_max_point_ct}")
print(f"Species name: {species[idx_max_point_ct]}")

api_for_species = API[idx_max_point_ct, :] # get API : reactions contribution to largest pointer

# Get only top 10 
top_n = 10
sorted_idx = np.argsort(np.abs(api_for_species.real))[::-1][:top_n]  # top 10 indices

# get the names of reactions 
reactions = gas.reaction_equations()
n_reactions = len(reactions)
reaction_labels = []

# fix latex issue
def fix_reaction_str(s):
    s = s.replace('<=>', r'$\leftrightarrow$')  # LaTeX arrow
    s = s.replace('=>', r'$\rightarrow$')
    s = s.replace('<=', r'$\leftarrow$')
    return s
    
# loop to get the reactions name properly
for i in range(2 * n_reactions):
    reac_idx = i % n_reactions
    direction = '(f)' if i < n_reactions else '(b)'
    fixed_str = fix_reaction_str(reactions[reac_idx])
    reaction_labels.append(f"{reac_idx+1}{direction}: {fixed_str}")
    
    
# Extract top values and labels
top_api_values = api_for_species.real[sorted_idx]
top_labels = [reaction_labels[i] for i in sorted_idx]


############   OPENFOAM  ####################################################
print("size Levec of: ",Levec_points_of.shape)
B_of = Levec_points_of[idx_point,:,:] # API for given time
ctk_of = CanteraThermoKinetics("mech/New_h_mech.yaml")

# set thermochemical state in ctk 
T_sel_of = T_foam[idx_of]  # from OF
Y_sel_of = np.array([species_foam[sp][idx_of] for sp in gas.species_names])
ctk_of.TPY = T_sel_of, pressure, Y_sel_of

Smat_of = ctk_of.generalized_Stoich_matrix_const_p() # get Stoich matrix
Rates_of = ctk_of.Rates_vector() # get Reaction rates vector
API_foam = csp.CSP_amplitude_participation_indices(B_of, Smat_of, Rates_of, normtype='abs') # abs : normalize so that sum of absolute values per species is 1

#print(f"Largest radical pointer species index: {idx_max_point_of}")
#print(f"Species name: {species[idx_max_point_of]}")

print("Largest radical pointers for each species :")
for i, sp in enumerate(species):
    print(f"{sp:>4} | Cantera: {rad_pointers_max_ct[0, i]:.4f} | OpenFOAM: {rad_pointers_max_of[0, i]:.4f}")

api_for_species_foam = API_foam[idx_max_point_of, :] # get API : reactions contribution to largest pointer
# Extract top values and labels
top_api_values_foam = api_for_species_foam.real[sorted_idx]

#=== plot API Cantera vs OF ===============================================================================================================
bar_width = 0.4
y_pos = np.arange(top_n)

plt.figure(figsize=(10, 6))
plt.barh(y_pos - bar_width/2, top_api_values, height=bar_width, color='navy', label='Cantera')
plt.barh(y_pos + bar_width/2, top_api_values_foam, height=bar_width, color='firebrick', label='OpenFOAM')

plt.yticks(y_pos, top_labels)
plt.xlabel('Participation Index (API)')
plt.title(
    rf"Top {top_n} Reactions for Radical pointer: {species[idx_max_point_ct]} (Cantera vs OpenFOAM) \\"
    rf" at Temperature $T \approx {T_cantera[idx_ct]:.1f}$ K", fontsize=12
)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().invert_yaxis()  # Highest value at top
plt.tight_layout()
plt.savefig(f"{output_dir}/API_Cantera_vs_OpenFOAM.png", dpi=300)
print(f"plotting successful, result saved in {output_dir}/API_Cantera_vs_OpenFOAM.png")
plt.show()





