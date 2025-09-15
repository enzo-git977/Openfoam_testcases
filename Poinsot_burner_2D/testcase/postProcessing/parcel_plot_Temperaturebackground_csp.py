print("Importing libraries")
# === Libraries ======================================================================================================
import os
import sys 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotter import CSPPlotter 

# === Paths =========================================================================================================
print("Constructing path to vtk file and Pycsp directories for plotting background ")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
pycsp_path = os.path.join(CURRENT_DIR, 'PyCSP')        # Joins to form path to PyCSP
sys.path.insert(0, pycsp_path)                         # Adds to sys.path so we can import from PyCSP
import PyCSP.Functions as csp                          # import Pycsp
from PyCSP.ThermoKinetics import CanteraThermoKinetics # import Pycsp
	      	
base_dir = os.path.join(CURRENT_DIR, "getEulerianFields")      	          # Set base_dir to the getEulerianFields directory
mech_dir = os.path.join(CURRENT_DIR,'mech','mechanism.yaml')  # mech file path
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results","Trajectory")	          # Directory where are stored output pictures
vtk_path = os.path.join(CURRENT_DIR, "cuttingPlanes", "0.04", "zMid.vtk") # vtk plane for plotting
os.makedirs(RESULTS_DIR, exist_ok=True)


# === Mechanism =====================================================================================================
gas = csp.CanteraCSP(mech_dir)
gas.constP = True
species = gas.species_names + ['T']
print("species = ", species)


# === HELPER FUNCTIONS ==============================================================================================
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
    
def is_WP_satisfied(p, T, Y):
    """
    Return True if max(lambda.real) > 0, False otherwise.
    """
    gas.TPY = T, p, Y
    lam, R, L, _ = gas.get_kernel()
    
    return lam.real.max() >= 1e-7

# === Acquire data ============================================================================================ 
# === CREATE INSTANCE ===
plotter = CSPPlotter(vtk_file=vtk_path, p=None, Y=None, x=None, x_indices=None, eigenvalues=None, vl=None)
# choice of parcel (parcel tried : 50 : move across all domain,55 : stuck at top boundary,)
target_parcel_id = 50
print(f"Tracking parcel number : {target_parcel_id} ")

# === INITIALISATION ===
x_positions = []
y_positions = []
T_positions = []
WP_flag_list= []
time_stamps = []

print("Accessing time folders to get parcel data...")
# === FIND TIME FOLDERS ===
start_time = 0.0705 # first folder with data 
end_time = 0.12

time_folders = sorted(
    [f for f in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, f)) and f.replace('.', '', 1).isdigit()],
    key=lambda x: float(x)
)

# Take some time folders not all of them 
filtered_time_folders = [
    f for f in time_folders if start_time <= float(f) <= end_time
]

# === LOOP OVER TIME STEPS ===
for time_str in filtered_time_folders:
    file_path = os.path.join(base_dir, time_str, 'airCloud.dat')
    

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
            # === EXCTRACT values === 
            x = parcel_rows.iloc[0]['X']
            y = parcel_rows.iloc[0]['Y']
            T = parcel_rows.iloc[0]['T']
            p = parcel_rows.iloc[0]['p']
            # === Extract species mass fractions ===
            Y_list = []
            for species in gas.species_names:
                if species in parcel_rows.columns:
                    Y_list.append(parcel_rows.iloc[0][species])
                else:
                    Y_list.append(0.0)
            Y = np.array(Y_list)
            WP_flag_list.append(is_WP_satisfied(p, T, Y))
            x_positions.append(x)
            y_positions.append(y)
            T_positions.append(T)
            time_stamps.append(float(time_str))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
# === CHECK IF PARCEL WAS FOUND ===
if not x_positions:
    print(f"Parcel ID {target_parcel_id} not found in any time folder.")
    sys.exit()
    
# === PLOT ============================================================================================

# === SET FIXED AXIS LIMITS BASED ON ALL DATA ===
fixed_xlim = (-0.030, 0.3) # zoom in order to see better the parcels
fixed_ylim = (-0.002, 0.052)

# Load the mesh once
x,y,z,connect,triang,mesh_data = plotter._load_mesh()

# === PLOTTING LOOP ===
for i in range(len(x_positions)):
    #plt.figure(figsize=(6, 5))
  
    fig, ax = plt.subplots(figsize=(12, 6))  
     
    # --- Load temperature field for this timestep ---
    time_str_folder = f"{time_stamps[i]:.6f}".rstrip('0').rstrip('.')
    vtk_file_t = os.path.join(CURRENT_DIR, "cuttingPlanes", time_str_folder, "zMid.vtk")
    plotter_t = CSPPlotter(vtk_file=vtk_file_t, p=None, Y=None, x=None, x_indices=None, eigenvalues=None, vl=None)
    
    # Load and set temperature field
    Temperature_array = plotter_t._load_scalar("T")
    if Temperature_array is not None:
        cf = ax.tripcolor(triang,Temperature_array,shading='gouraud',cmap='inferno',vmin=280,vmax=1900)
        fig.colorbar(cf, ax=ax, label="Temperature [K]",orientation='horizontal', pad=0.15)
        
        
    else:
        print(f"No temperature field found at time {time_stamps[i]:.4f}")
          
    # Plot previous positions of parcel in grey
    if i > 0:
        ax.plot(x_positions[:i+1], y_positions[:i+1], 'o-', color='grey', alpha=0.3)
    
    # Compute shifted arrow positions for segments up to i
        sx, sy = shifted_points(x_positions[:i+1], y_positions[:i+1], shift_fraction=0.1)

    # Plot only first and last arrow 
        if len(sx) >= 1:
            ax.plot(sx[0], sy[0], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)
            if len(sx) > 1:
                ax.plot(sx[-1], sy[-1], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)

    # Plot current position in blue
    ax.plot(x_positions[i], y_positions[i], 'o', color='blue',label=f'Parcel ID: {target_parcel_id}')
    
    
    ax.set_aspect('equal', "box") #for some reason it leads to the axis changing for every timestep so comment
    
    # Fix axes to constant limits
    ax.set_xlim(fixed_xlim)
    ax.set_ylim(fixed_ylim)
    ax.legend(
    loc='center left', 
    bbox_to_anchor=(0.76, 0.9),
    fontsize=10,           # small font size
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
    f"T = {T_positions[i]:.4f} K \n{flag_text}",
    xy=(x_positions[i], y_positions[i]),              # point (target of arrow)
    xytext=(x_positions[i], y_positions[i] + 0.005),  # text location (above point)
    fontsize=8,
    ha='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6, lw=1)
    )

    # Styling 
    #ax.set_title(f"Parcel Trajectory - t = {time_stamps[i]:.4f} s")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Save the image
    #plt.tight_layout()
    time_str = f"{time_stamps[i]:.4f}".replace('.', '_')
    fig.savefig(os.path.join(RESULTS_DIR,f"parcel_{time_str}.png"), dpi=150)
    plt.close(fig)
    print(f"Plotting successful for t={time_str}")

print(f"All plots saved in {RESULTS_DIR}")

    

