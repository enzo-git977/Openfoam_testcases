print("Starting execution")
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import integrate
from scipy.interpolate import interp1d

# Latex format
plt.rc('text',usetex='True')
plt.rc('font',family='serif',size=21)

# === Paths =======================================================================================================
current_dir = os.getcwd()  # Gets current working directory
Cantera_path = 'Cantera_solution_Stoich/data_new_mech/cantera_profile.csv'
time = "0.01"
OF_path = os.path.join(current_dir, "Openfoam_solution_Stoich", "postProcessing", "axialLine", time, "line1.xy")
output_dir = 'figures/'
os.makedirs(output_dir, exist_ok=True)

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
    'OH*': df_cantera['OH*'].values,
    'HE': df_cantera['HE'].values,
}
#['H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']

print("Loaded Cantera data")

# === Load OpenFOAM Data =========================================================================================
foam_data = pd.read_csv(OF_path, comment='#', header=None,sep='\s+')
foam_data.columns = ['x','T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']

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
    'OH*': foam_data['OH*'].values,
    'HE': foam_data['HE'].values,
}

print("Loaded OpenFOAM data")

# === Shifting so that both plot can be compared ==================================================================
T_target = 1500
idx_peak_cantera = np.argmin(np.abs(T_cantera - T_target))
idx_peak_foam = np.argmin(np.abs(T_foam - T_target))
x_peak_cantera = x_cantera[idx_peak_cantera]
x_peak_foam = x_foam[idx_peak_foam]
# Compute the shift to apply to OpenFOAM to align with Cantera
shift = x_peak_cantera - x_peak_foam
print(f" Aligning peaks: shift OpenFOAM by {shift:.3f} mm")

# Apply shift
x_foam_shifted = x_foam + shift

# === Compute error between Cantera and OF =======================================================================
dict_error = {}
# position 
x_min= 10  # mm
x_max = 40 # mm
mask_of = (x_foam_shifted >= x_min) & (x_foam_shifted <= x_max)

# species : 
#species_all = ['T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']
species_all = ['T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'OH*']

for sp in species_all:
    if sp == 'T':
        y_ct = T_cantera
        y_of = T_foam
    else:
        y_ct = species_cantera[sp]
        y_of = species_foam[sp]
        
    # Interpolate Cantera data on OF grid
    interp_ct = interp1d(x_cantera, y_ct, bounds_error=False, fill_value="extrapolate")
    y_ct_interp = interp_ct(x_foam_shifted)
    # error calculation
    phi_cant = integrate.simpson(y_ct_interp[mask_of], x_foam_shifted[mask_of]) # simpson integral
    phi_of = integrate.simpson(y_of[mask_of], x_foam_shifted[mask_of])
    error = (np.abs(phi_of- phi_cant)/(phi_cant+1e-12))  # +1e-12 : avoid 0 division
    dict_error[sp] = error
    
species = list(dict_error.keys())

def latex_label(sp):
    # A simple way: insert underscores before digits
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
#plt.title(f"Integral error Cantera vs Openfoam ({x_min} to {x_max} mm)")
plt.grid(axis='y',linestyle='--', alpha=0.7)
plt.tight_layout()
#plt.savefig(f"{output_dir}/OFvsCantera_errors.png", dpi=300)
plt.savefig(f"{output_dir}/OFvsCantera_errors.pdf", dpi=300)
plt.close()


# === Plot Temperature ============================================================================================
print("Plotting Temperature")

plt.figure(figsize=(8, 5))
plt.plot(x_cantera, T_cantera, label='Cantera', color='firebrick')
plt.plot(x_foam_shifted[::5], T_foam[::5], label='OpenFOAM', color='navy',marker= 'o', linestyle='none') # plot every 5 points
plt.xlabel('Position [mm]')
plt.ylabel('Temperature [K]')
plt.title('Temperature Profile: Cantera vs OpenFOAM')
plt.legend(fontsize=12, frameon=False)
plt.grid(False)
plt.tight_layout()
#plt.savefig(f"{output_dir}/OFvsCantera_Temperature.png", dpi=300)
plt.savefig(f"{output_dir}/OFvsCantera_Temperature.pdf", dpi=300)
plt.close()

# === Plot Species Mass Fractions ================================================================================
print("Plotting Species Mass Fractions")

#['H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']
species_to_plot = ['H','O','H2','O2','OH','H2O','N2','N','NO','HO2','H2O2','OH*']
num_species = len(species_to_plot)
n_row=3
n_col = 4

fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4.5, n_row * 3), sharex=True)
axs = axs.flatten()

for i, sp in enumerate(species_to_plot):
    ax = axs[i]
    ax.plot(x_cantera, species_cantera[sp], label='Cantera', color='firebrick')
    ax.plot(x_foam_shifted, species_foam[sp], label='OpenFOAM', color='navy', linestyle='--')
    ax.set_ylabel(rf"$Y_{{\mathrm{{{sp}}}}}$")
    ax.grid(False)

for ax in axs[-n_col:]:
    ax.set_xlabel('Position [mm]')
    
# Shared legend on top
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='upper right', fontsize=12, frameon=False)
    
fig.suptitle('Species Mass Fraction Profiles: Cantera vs OpenFOAM', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
#plt.savefig(f"{output_dir}/OFvsCantera_MassSpecies.png", dpi=300)
plt.savefig(f"{output_dir}/OFvsCantera_MassSpecies.pdf", dpi=300)
plt.close()

print("Plots saved in 'figures/' folder.")

