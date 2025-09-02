print("Starting execution")
import numpy as np 
import pandas as pd
import os

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

print("Loaded Cantera data")

# === Load OpenFOAM Data =========================================================================================
foam_data = pd.read_csv(OF_path, comment='#', header=None,sep='\s+')
foam_data.columns = ['x','T','H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']

x_foam = foam_data['x'].values * 1000  # convert m to mm to match Cantera
T_foam = foam_data['T'].values

print("Loaded OpenFOAM data")

# === Search closest match of target OF Temperature to Cantera ======================================================
tol = 0.2 # temperature tolerance in K
matches = []
seen = set()

for i, T_c in enumerate(T_cantera):
    idx_closest = np.argmin(np.abs(T_foam - T_c))
    T_f = T_foam[idx_closest]
    delta_T = np.abs(T_f - T_c)
    
    # Round temperature to avoid floating-point noise
    T_c_rounded = round(T_c, 2)
    
    if delta_T <= tol and abs(T_c_rounded < 2000) and T_c_rounded not in seen:
        seen.add(T_c_rounded)
        matches.append({
            'x_cantera': x_cantera[i],
            'T_cantera': T_c,
            'x_foam': x_foam[idx_closest],
            'T_foam': T_f,
            'delta_T': delta_T
        })
for m in matches:
    print(f"{m['T_cantera']:12.2f}  | {m['T_foam']:10.2f} | {m['delta_T']:8.2f}")
    
   


    





