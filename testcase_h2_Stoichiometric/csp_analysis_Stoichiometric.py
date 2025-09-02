print("Starting execution")
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

# === Paths =======================================================================================================
csv_path = 'Cantera_solution_Stoich/data_Stoichiometric/cantera_states.csv'
npz_path = 'Cantera_solution_Stoich/data_Stoichiometric/csp_data.npz'
output_dir = 'figures_Stoichiometric/'
os.makedirs(output_dir, exist_ok=True)

# === Load simulation data from Cantera  ==========================================================================
with open(csv_path) as f:
    header = f.readline().lstrip('#').strip().split(',')

df_states = pd.read_csv(csv_path, comment='#', header=None)
df_states.columns = header

# Extract values
time = df_states['time'].values
T = df_states['T'].values
species_cantera = {
    'H': df_states['H'].values,
    'O': df_states['O'].values,
    'H2': df_states['H2'].values,
    'O2': df_states['O2'].values,
    'OH': df_states['OH'].values,
    'H2O': df_states['H2O'].values,
    'N2': df_states['N2'].values,
    'N': df_states['N'].values,
    'NO': df_states['NO'].values,
    'HO2': df_states['HO2'].values,
    'H2O2': df_states['H2O2'].values,
    'OH*': df_states['OH*'].values,
    'HE': df_states['HE'].values,
}

# === Load CSP data ===
data = np.load(npz_path)
evals = data['evals']
Revec = data['Revec']
Levec = data['Levec']
fvec = data['fvec']

#print("time ct=", len(time))
print("last time ct=", time[-1]) #last time = 0.00099 whereas last time of = 0.001 and time_foam[-2] = time_ct[-1]

print("Loaded thermo and CSP data from Cantera")

#======== OpenFoam SIMULATION ==============================================================================
# Load OpenFOAM data
foam_data = pd.read_csv('data_Foam_Stoichiometric.csv', comment='#', header=None)
foam_data.columns = ['time','O2','AR','O','H2O2','HO2','T','H2O','NO','N2','OH','H2','OH*','HE','N','H']

foam_data = foam_data.iloc[:-1, :]  # remove last row from dataframe  to match cantera

# Time and temperature
time_foam = foam_data['time'].values # .values : return numpy array
T_foam = foam_data['T'].values

# Species mass fractions from OpenFOAM
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
    'OH*': foam_data['OH*'].values,
    'HE': foam_data['HE'].values,
}

print("Loaded thermo data from Foam")

#print("time of=", len(time_foam))
print("last time of=", time_foam[-1])

#======== OpenFoam CSP_analysis ==============================================================================
# Mechanism
gas_foam = csp.CanteraCSP('mech/New_h_mech.yaml')
gas_foam.constP = True
species = gas_foam.species_names + ['T']
print("species = ", species)
#print(">>> Jacobian type set to:", gas_foam.jacobiantype) # just to check
#print(">>> nv (number of variables):", gas_foam.nv)
# Initialize containers
evals_foam = []
Revec_foam = []
Levec_foam = []
fvec_foam = []
P_foam = 2*ct.one_atm
gas_foam.TP = 1000, P_foam
# get eigenvalues and eigenvectors for every timestep
for i in range(len(time_foam)): 
    T_foam_i = T_foam[i]
    Y_foam_i = np.zeros(gas_foam.n_species)
    # Fill species mass fractions
    for j, sp in enumerate(gas_foam.species_names):
        if sp in foam_data.columns:
            Y_foam_i[j] = foam_data[sp][i]
    # Set Cantera gas state to update kernel
    gas_foam.TPY = T_foam_i, P_foam, Y_foam_i
    lam_foam, R_foam, L_foam, f_foam = gas_foam.get_kernel()
    evals_foam.append(lam_foam)
    Revec_foam.append(R_foam)
    Levec_foam.append(L_foam)
    fvec_foam.append(f_foam)
# convert to array
evals_foam = np.array(evals_foam)
Revec_foam = np.array(Revec_foam)
Levec_foam = np.array(Levec_foam)
fvec_foam = np.array(fvec_foam)
print("Eigenvalues and eigenvectors for Foam simulation obtained")    

# just for testing 
"""
# === Plot Temperature ============================================================================================
print("Plotting Temperature...")
plt.figure(figsize=(10, 6))
plt.plot(time, T, label='Cantera', color='firebrick')
plt.plot(time_foam, T_foam, label='OpenFOAM', color='navy',marker= 'o', linestyle='none') # plot every 5 points
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.title('Temperature Profile: Cantera vs OpenFOAM')
plt.legend(fontsize=12, frameon=False)
plt.grid(False)
plt.xlim(0, 0.001)
plt.savefig(f"{output_dir}/OFvsCantera_Temperature.png", dpi=300)
plt.close()

# === Plot Species Mass Fractions ================================================================================
print("Plotting Species Mass Fractions...")

#['H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']
species_to_plot = ['H','O','H2','O2','OH','H2O','N2','N','NO','HO2','H2O2','OH*']
num_species = len(species_to_plot)
n_row=3
n_col = 4

fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4.5, n_row * 3), sharex=True)
axs = axs.flatten()

for i, sp in enumerate(species_to_plot):
    ax = axs[i]
    ax.plot(time, species_cantera[sp], label='Cantera', color='firebrick')
    ax.plot(time_foam, species_foam[sp], label='OpenFOAM', color='navy', linestyle='--')
    ax.set_ylabel(rf"$Y_{{\mathrm{{{sp}}}}}$")
    ax.grid(False)

for ax in axs[-n_col:]:
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 0.001)
    
# Shared legend on top
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='upper right', fontsize=12, frameon=False)
    
fig.suptitle('Species Mass Fraction Profiles: Cantera vs OpenFOAM', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/OFvsCantera_MassSpecies.png", dpi=300)
plt.close()

print(f"Plots saved in '{output_dir}/' folder.")
"""
"""
# === Plot eigenvalue magnitudes ==============================================================
# Compute log-scaled eigenvalues
logevals_ct = np.clip(np.log10(1.0 + np.abs(evals.real)), 0, 100) * np.sign(evals.real)
logevals_of = np.clip(np.log10(1.0 + np.abs(evals_foam.real)), 0, 100) * np.sign(evals_foam.real)

# convert to array
time_ct = np.array(time)
time_of = np.array(time_foam)

print(len(time_ct), logevals_ct.shape[0])  # Should match
print(len(time_of), logevals_of.shape[0])  # Should match


print("First 10 eigenvalues at first time step (Cantera):", evals[0, :10])
print("First 10 eigenvalues at first time step (OpenFOAM):", evals_foam[0, :10])



plt.figure(figsize=(10, 5))

# Plot Cantera eigenvalues
for idx in range(logevals_ct.shape[1]):
    plt.plot(time_ct, logevals_ct[:, idx], 'k.', markersize=2, label='Cantera' if idx == 0 else "")

# Plot OpenFOAM eigenvalues
for idx in range(logevals_of.shape[1]):
    plt.plot(time_of, logevals_of[:, idx], 'r.', markersize=2, label='OpenFOAM' if idx == 0 else "")

plt.xlabel('Time [s]')
plt.ylabel(r'$\log_{10}(|\mathrm{Re}(\lambda)| + 1) \cdot \mathrm{sign}(\mathrm{Re}(\lambda))$')
plt.title("Eigenvalue Comparison: Cantera vs. OpenFOAM")
plt.xlim([0., 0.001])
plt.ylim([-9, 6])
plt.grid(False)
plt.legend(loc='upper right',fontsize=12, frameon=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/Eigenvalues_Cantera_vs_OpenFOAM.png', dpi=500)
plt.show()
print(f"Eigenvalue comparison plot saved to {output_dir}/Eigenvalues_Cantera_vs_OpenFOAM.png")
"""


# === CSP Analysis of 6 points for OpenFOAM ==========================================================
# Find closest indices in the OpenFOAM time vector
target_times = [0.00015,0.0002, 0.00027, 0.0002879, 0.00035, 0.0005]
n_points = len(target_times)
indices_of = [np.argmin(np.abs(time_foam - t)) for t in target_times]


# Extract corresponding CSP data from OpenFOAM
evals_points_of = evals_foam[indices_of]        # (n_points, n_species)
Revec_points_of = Revec_foam[indices_of]        # (n_points, n_species, n_species)
Levec_points_of = Levec_foam[indices_of]        # (n_points, n_species, n_species)

# Compute radical pointers for OpenFOAM
rad_all_of = []
for i in range(n_points):
    rad_i = csp.CSP_pointers(Revec_points_of[i], Levec_points_of[i])
    rad_all_of.append(rad_i)

rad_all_of = np.array(rad_all_of)

# Max real eigenvalue indices
max_idx_of = np.argmax(evals_points_of.real, axis=1)

# Extract radical pointers for max eigenvalue (OpenFOAM)
rad_pointers_max_of = np.array([rad_all_of[i, idx] for i, idx in enumerate(max_idx_of)])
rad_pointers_max_of = rad_pointers_max_of / np.sum(np.abs(rad_pointers_max_of), axis=1, keepdims=True) # Normalize per species (so they sum to 1 per point)
rad_pointers_max_of = np.abs(rad_pointers_max_of.real)

# === CSP Analysis of 6 points for Cantera ==============================================================
# Find closest indices in the Cantera time vector
aligned_times = time_foam[indices_of]  # exact OpenFOAM time points
indices_ct = [np.argmin(np.abs(time - t)) for t in aligned_times] # match OF times with Cantera times so that we plot the same thing


print("Extracting CSP data (Cantera) at the following time points:")
for t, idx in zip(target_times, indices_ct):
    #print(f"Time {t:.8f} s, Closest actual time: {time[idx]:.8f} s, index: {idx}")
    print(f"Time (ct) {time[idx]:.8f} s | Time (of) {time_foam[idx]:.8f} s")
    print(f"Temperature (ct) = {T[idx]} | Temperature (of) = {T_foam[idx]}")
    
   

# Exctract according csp data
evals_points = evals[indices_ct]  # Shape: (n_points, n_species)
Revec_points = Revec[indices_ct]  # Shape: (n_points, n_species, n_species)
Levec_points = Levec[indices_ct]  # Shape: (n_points, n_species, n_species)
# Compute radical pointers for all modes at each point
rad_all=[]
for i in range(n_points):
    rad_i = csp.CSP_pointers(Revec_points[i,:,:], Levec_points[i,:,:])  
    rad_all.append(rad_i)
rad_all  =np.array(rad_all)
# Index of max eigenvalue (largest real part) at each point
max_idx = np.argmax(evals_points.real, axis=1)  # Shape: (n_points,) 

# Extract radical pointers corresponding to max eigenvalue
rad_pointers_max = np.array([rad_all[i, idx] for i, idx in enumerate(max_idx)])
rad_pointers_max = rad_pointers_max / np.sum(np.abs(rad_pointers_max), axis=1, keepdims=True) # Normalize per species (so they sum to 1 per point)
rad_pointers_max = np.abs(rad_pointers_max.real)

#print("rad_pointers_max shape:", rad_pointers_max.shape) # just to check if everything went as expected 

# === Combined Radical Pointers Plot for Cantera and OpenFOAM =======================================
x = np.arange(len(species))  # bar positions for species
width = 0.3                  # narrower bars for side-by-side comparison

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)

for i, ax in enumerate(axs.flat):
    # Offset bars for comparison
    ax.bar(x - width/2, rad_pointers_max[i], width, label='Cantera', color='firebrick')
    ax.bar(x + width/2, rad_pointers_max_of[i], width, label='OpenFOAM', color='navy')

    ax.set_xticks(x)
    ax.set_xticklabels(species, ha='right',fontsize=7)
    ax.set_ylabel("Radical Pointer Value" if i % 3 == 0 else "")
    ax.set_title(rf"$t \approx {target_times[i]:.5f}~\mathrm{{s}}$")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    if i == 0:
        ax.legend(fontsize=12, frameon=False)

plt.suptitle("Radical Pointers (Max Eigenvalue) at Selected Times\nComparison: Cantera vs OpenFOAM", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.ylim(0,1)
plt.savefig(f"{output_dir}/radical_pointer_Cantera_vs_OpenFOAM.png", dpi=150)
plt.show()
#"""


#"""
#===== III ======#
#===== Participation indexes at given time for Cantera =============================================
# target time  0.00027
target_time = 0.00015000

# Find index in time array (Cantera or OpenFOAM)
idx_of = np.argmin(np.abs(time_foam - target_time))  # OpenFOAM*
idx_ct = np.argmin(np.abs(time - target_time)) # cantera

# 
print(f"OpenFOAM time: {time_foam[idx_of]}")
print(f"Cantera time:  {time[idx_ct]}")

idx_lambda = evals[idx_ct].real.argsort()[::-1] # sort eignevalues from largest real part to smaller
dominant_mode = idx_lambda[0] # dominant mode
rad = csp.CSP_pointers(Revec[idx_ct,:,:], Levec[idx_ct,:,:])  # compute rad_pointers
rad_dominant = rad[dominant_mode, :] # Extract radical pointer vector for dominant mode (shape: n_species,)
rad_dominant_norm = np.abs(rad_dominant) / np.sum(np.abs(rad_dominant)) # normalize
largest_pointer = rad_dominant_norm.argmax()  # largest pointer

print("size Levec : ",Levec.shape)
B = Levec[idx_ct,:,:] # API for given time
ctk = CanteraThermoKinetics("mech/New_h_mech.yaml")
# set thermochemical state in ctk 
T_sel = T[idx_ct]  # from Cantera
Y_sel = df_states.loc[df_states.index[idx_ct], gas_foam.species_names].values
ctk.TPY = T_sel, P_foam, Y_sel

Smat = ctk.generalized_Stoich_matrix_const_p() # get Stoich matrix
Rates = ctk.Rates_vector() # get Reaction rates vector
API = csp.CSP_amplitude_participation_indices(B, Smat, Rates, normtype='abs') # abs : normalize so that sum of absolute values per species is 1

print(f"Largest radical pointer species index: {largest_pointer}")
print(f"Species name: {gas_foam.species_names[largest_pointer]}")

api_for_species = API[largest_pointer, :] # get API : reactions contribution to largest pointer

# Get only top 10 
top_n = 10
sorted_idx = np.argsort(np.abs(api_for_species.real))[::-1][:top_n]  # top 10 indices

# get the names of reactions 
reactions = gas_foam.reaction_equations()
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

#===== Participation indexes at last timestep for OopenFoam =============================================
idx_lambda_foam = evals_foam[idx_of].real.argsort()[::-1] # sort eignevalues from largest real part to smaller
dominant_mode_foam = idx_lambda_foam[0] # dominant mode
rad_of = csp.CSP_pointers(Revec_foam[idx_of,:,:], Levec_foam[idx_of,:,:])  # compute rad_pointers
rad_dominant_foam = rad_of[dominant_mode_foam, :] # Extract radical pointer vector for dominant mode (shape: n_species,)
rad_dominant_foam = np.abs(rad_dominant_foam) / np.sum(np.abs(rad_dominant_foam)) # normalize
largest_pointer_foam = rad_dominant_foam.argmax()  # largest pointer


print("size Levec_foam : ",Levec_foam.shape)
B_foam = Levec_foam[idx_of,:,:] # API for last timestep
ctk_foam = CanteraThermoKinetics("mech/New_h_mech.yaml")
# set thermochemical state in ctk 
T_sel_foam = T_foam[idx_of]
Y_sel_foam = foam_data.loc[foam_data.index[idx_of], gas_foam.species_names].values
ctk_foam.TPY = T_sel_foam, P_foam, Y_sel_foam

Smat_foam = ctk_foam.generalized_Stoich_matrix_const_p() # get Stoich matrix
Rates_foam = ctk_foam.Rates_vector() # get Reaction rates vector
API_foam = csp.CSP_amplitude_participation_indices(B_foam, Smat_foam, Rates_foam, normtype='abs') # abs : normalize so that sum of absolute values per species is 1


print("Largest radical pointers for each species :")
for i, sp in enumerate(species):
    print(f"{sp:>4} | Cantera: {rad_pointers_max[0, i]:.4f} | OpenFOAM: {rad_pointers_max_of[0, i]:.4f}")

print("T_ct = ",T_sel)
print("T_of = ",T_sel_foam)

api_for_species_foam = API_foam[largest_pointer_foam, :] # get API : reactions contribution to largest pointer
    
# Extract top values and labels
top_api_values_foam = api_for_species_foam.real[sorted_idx]

# plot API Cantera vs OF ====================================================================
bar_width = 0.4
y_pos = np.arange(top_n)

plt.figure(figsize=(10, 6))
plt.barh(y_pos - bar_width/2, top_api_values, height=bar_width, color='firebrick', label='Cantera')
plt.barh(y_pos + bar_width/2, top_api_values_foam, height=bar_width, color='navy', label='OpenFOAM')

plt.yticks(y_pos, top_labels)
plt.xlabel('Participation Index (API)')
plt.title(rf'Top {top_n} Reactions for Radical pointer: {gas_foam.species_names[largest_pointer]} (Cantera vs OpenFOAM) \\ at time $t \approx {target_time:.5f}$ s',fontsize=12)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().invert_yaxis()  # Highest value at top
plt.tight_layout()
plt.savefig(f"{output_dir}/API_Cantera_vs_OpenFOAM.png", dpi=150)
plt.show()
#"""
"""
# plot only one ====================================================================
plt.figure(figsize=(8, 6))
bars = plt.barh(range(top_n), top_api_values, color='steelblue')
plt.yticks(range(top_n), top_labels)
plt.xlabel('Participation Index (API)')
plt.title(f'Top {top_n} Reactions for Species: {gas_foam.species_names[largest_pointer]}')
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.show()
"""

# =======






