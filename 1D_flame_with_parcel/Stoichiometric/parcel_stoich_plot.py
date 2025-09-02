print("Importing libraries")
import os
import sys 
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct
import numpy as np

# === Paths ===================================================================================================================
current_dir = os.getcwd()  # Gets current working directory
time_folders_path = os.path.join(current_dir, "Openfoam_solution-stoich-coarse", "postProcessing", "getEulerianFields") 
time = "0.04"
OF_path = os.path.join(current_dir, "Openfoam_solution-stoich-coarse", "postProcessing", "axialLine", time, "line1.xy")
fig_dir = "figures_Stoich"
os.makedirs(fig_dir, exist_ok=True)

# import Pycsp
pycsp_path = os.path.abspath(os.path.join(current_dir, "..","PyCSP"))
sys.path.insert(0, pycsp_path)  # Adds to sys.path so we can import from PyCSP

import PyCSP.Functions as csp
from PyCSP.ThermoKinetics import CanteraThermoKinetics

# === HELPER FUNCTION  ========================================================================================================
def shifted_points(x, y, shift_fraction=0.1):
    shifted_x = []
    shifted_y = []
    for j in range(len(x) - 1):
        dx = x[j+1] - x[j]
        dy = y[j+1] - y[j]
        # Point shifted backward from the end point by shift_fraction
        shifted_x.append(x[j+1] - dx * shift_fraction)
        shifted_y.append(y[j+1] - dy * shift_fraction)
    return shifted_x, shifted_y

# === Load Mechanism ==========================================================================================================
gas = csp.CanteraCSP('Openfoam_solution-stoich-coarse/mech/hydrogen.yaml')
phi = 1 # stoich
gas.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
pressure = ct.one_atm
T_in = 300.0
gas.constP = True
gas.TP = T_in, pressure
species = gas.species_names + ['T']
print("species = ", species)


# === Load Temperature of the whole domain ====================================================================================
foam_data = pd.read_csv(OF_path, comment='#', header=None,sep='\s+')
foam_data.columns = ['x','T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']

x_foam = foam_data['x'].values * 1000  # convert m to mm to match Cantera
T_foam = foam_data['T'].values

# === Get Parcel's thermodynamic data =========================================================================================
target_parcel_id = 15 # tried: 3,10
print(f"Tracking parcel number : {target_parcel_id} ")

# === INITIALISATION ===
x_list = []
y_list = []
T_list = []
Mfrac_species_list = []
WP_flag_list= []
time_stamps = []

print("Collecting data from time folders...")
# === FIND TIME FOLDERS ===
time_folders = sorted(
    [f for f in os.listdir(time_folders_path)
     if os.path.isdir(os.path.join(time_folders_path, f)) and f.replace('.', '', 1).isdigit()],
    key=lambda x: float(x)
)

# === LOOP OVER TIME STEPS ===
for time_str in time_folders:
    file_path = os.path.join(time_folders_path, time_str, 'Cloud.dat')

    if not os.path.isfile(file_path):
        continue

    try:
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        # === CREATE DATAFRAME ===
        data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, engine='python')

        # === EXCTRACT HEADER ===
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header = line.lstrip('#').strip().split()
                break

        data.columns = header  
        #print(f"Extracted header: {header}") # for debug

        if data.empty:
            continue
        
        parcel_rows = data[data['PARCEL_ID'] == target_parcel_id] # we want the info of the target_parcel_id
	
        if not parcel_rows.empty:
            # === Exctract x and T values === 
            x = parcel_rows.iloc[0]['X']*1000 # mm
            y = parcel_rows.iloc[0]['Y']
            T = parcel_rows.iloc[0]['T']
            # === Extract species mass fractions ===
            Y_list = []
            for species in gas.species_names:
                if species in parcel_rows.columns:
                    #print("Y =", species)
                    Y_list.append(parcel_rows.iloc[0][species])
                else:
                    Y_list.append(0.0)
            Y_t = np.array(Y_list)
            
            # Append to list
            x_list.append(x)
            y_list.append(y)
            T_list.append(T)
            Mfrac_species_list.append(Y_t)
            time_stamps.append(float(time_str))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
# === CHECK IF PARCEL WAS FOUND ===
if not x_list:
    print(f"Parcel ID {target_parcel_id} not found in any time folder.")
    sys.exit()
    
# === CSP Computations ========================================================================================================
print("Computing eigenvalues...") 
# Initialize containers
evals_parcel = []
Revec_parcel = []
Levec_parcel = []

# get eigenvalues and eigenvectors for every timestep
for i in range(len(x_list)):
    T_i = T_list[i]
    Y_i = Mfrac_species_list[i]
    
    # Set Cantera gas state to update kernel
    gas.TPY = T_i, pressure, Y_i
    lam_parcel, R_parcel, L_parcel, f_parcel = gas.get_kernel()
    evals_parcel.append(lam_parcel)
    Revec_parcel.append(R_parcel)
    Levec_parcel.append(L_parcel)
    
# convert to array
evals_parcel = np.array(evals_parcel)
Revec_parcel = np.array(Revec_parcel)
Levec_parcel = np.array(Levec_parcel)

print("Eigenvalues and eigenvectors obtained for all timesteps") 

# Compute the explosive ones
for i in range(len(x_list)):
    if (max(evals_parcel[i].real) >= 1e-7):
                wherePositive = 1.0
                WP_flag_list.append(wherePositive)
    else:
        wherePositive = 0.0
        WP_flag_list.append(wherePositive)
            
print("Plotting...")           
# === PLOTTING LOOP ===========================================================================================================
plot_every_dt = 1e-3  # plot every 0.0001 s
last_plot_time = time_stamps[0]

for i in range(len(time_stamps)):
    dt = time_stamps[i] - last_plot_time
    if dt < plot_every_dt and i != len(time_stamps) - 1:
        continue  
        
    last_plot_time = time_stamps[i]
    # start plotting
    fig, ax = plt.subplots(figsize=(8, 4)) 
    ax2 = ax.twinx()  # second y-axis 

    # Plot temperature
    ax2.plot(x_foam, T_foam, '-', color='firebrick', label='Temperature')
    
    # Plot previous positions in grey
    if i > 0:
        skip = 1
        indices_skip = np.arange(0,i,skip) # plot only 10 of them 
        ax.plot(np.array(x_list)[indices_skip], np.array(y_list)[indices_skip], 'o', color='grey', alpha=0.3)
    
    # Compute shifted arrow positions for segments up to i
        sx, sy = shifted_points(x_list[:i+1], y_list[:i+1], shift_fraction=0.1)

    # Plot only first  arrow 
        if len(sx) >= 1:
            ax.plot(sx[0], y_list[0], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)
         
    # Plot current position in blue
    ax.plot(x_list[i], y_list[i], 'o', color='blue',label=f'Parcel ID: {target_parcel_id}')
    
    #ax.set_aspect('equal', "box") #for some reason it leads to the axis changing for every timestep so comment
    
    # Fix axes to constant limits
    ax.legend(
    loc='center left', 
    bbox_to_anchor=(0.76, 0.9),
    fontsize=8,           # small font size
    handlelength=1.5,     # length of the legend line (shorter than default)
    handleheight=0.5,     # height of the legend handle
    handletextpad=0.4,    # space between handle and text
    markerscale=0.5,      # scale down marker size
    borderaxespad=0       # reduce padding between legend and axes
    )
    
    
    # Flag to point out if species are explosive
    flag_text = "Explosive" if bool(WP_flag_list[i]) else "Not Explosive"
    
    # Annotate with info box and arrow 
    ax.annotate(
    f"T = {T_list[i]:.4f} K \n{flag_text}",
    xy=(x_list[i], y_list[i]),              # point (target of arrow)
    xytext=(x_list[i], y_list[i] + 0.005),  # text location (above point)
    fontsize=8,
    ha='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6, lw=1)
    )

    # Styling 
    ax.set_title(f"Parcel Trajectory - t = {time_stamps[i]:.4f} s")
    ax.set_ylim(-0.01, 0.02)  # Narrow y-axis to make it look 1D
    ax.set_yticks([]) 
    ax.set_xlabel("Position [mm]")
    ax2.set_ylabel("Temperature [K]")
    ax2.legend(loc="lower right")
    
    # Save the image
    #plt.tight_layout()
    fig.savefig(f"{fig_dir}/parcel_{i:08d}.png", dpi=150)
    plt.close(fig)
    print(f"Plotting successful for t={time_stamps[i]:.8f}")
print(f"Pictures.png saves in {fig_dir}")
    

