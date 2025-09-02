print("Starting execution")
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy import integrate
from scipy.interpolate import interp1d

current_dir = os.getcwd()  # Gets current working directory
pycsp_path = os.path.join(current_dir, 'PyCSP')  # Joins to form path to PyCSP
sys.path.insert(0, pycsp_path)  # Adds to sys.path so we can import from PyCSP

import PyCSP.Functions as csp
from PyCSP.ThermoKinetics import CanteraThermoKinetics

# Latex format
plt.rc('text',usetex='True')
plt.rc('font',family='serif',size=21)

# === Paths =======================================================================================================
csv_path = 'Cantera_solution/data_Lean/cantera_states.csv'
npz_path = 'Cantera_solution/data_Lean/csp_data.npz'
output_dir = 'figures_Lean/'
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

#======== Load simulation data from OpenFoam ==============================================================================
# Load OpenFOAM data
foam_data = pd.read_csv('data_Foam_Lean.csv', comment='#', header=None)
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


# === Compute error between Cantera and OF =======================================================================
dict_error = {}
# choose boundaries 
t_min= 0.0002 # s
t_max = 0.0006 # s
mask_of = (time_foam >= t_min) & (time_foam <= t_max)

# species : 
#species_all = ['T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']
species_all = ['T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'OH*','HE']

for sp in species_all:
    if sp == 'T':
        y_ct = T
        y_of = T_foam
    else:
        y_ct = species_cantera[sp]
        y_of = species_foam[sp]
        
    # Interpolate Cantera data on OF grid
    interp_ct = interp1d(time, y_ct, bounds_error=False, fill_value="extrapolate")
    y_ct_interp = interp_ct(time_foam)
    # error calculation
    phi_cant = integrate.simpson(y_ct_interp[mask_of], time_foam[mask_of]) # simpson integral
    phi_of = integrate.simpson(y_of[mask_of], time_foam[mask_of])
    error = (np.abs(phi_of- phi_cant)/(phi_cant+1e-12))  # +1e-12 : avoid 0 division
    dict_error[sp] = error
    
species = list(dict_error.keys())

def latex_label(sp):
    # insert underscores before digits
    import re
    # replace digits with _digit
    sp_latex = re.sub(r'(\d+)', r'_\1', sp)
    return rf"${sp_latex}$"

labels_latex = [latex_label(sp) for sp in species]

    
print("Plotting barchart of integral error")
# plot using barchart
plt.figure(figsize=(12,5))
plt.bar(species,dict_error.values(),color='black')
plt.ylabel("Relative integral error")
plt.title(f"Integral error Cantera vs Openfoam ({t_min} to {t_max} s)")
plt.grid(axis='y',linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/OFvsCantera_errors.png", dpi=300)
plt.close()
