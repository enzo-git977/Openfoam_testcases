"""
Wrapper script to run `Get_eigen_vtk.py`
for all time folders in `getEulerianFields`.  
"""

import os
import subprocess

# Paths
current_dir = os.getcwd()
folder_dir = os.path.join(current_dir, "getEulerianFields")
script_path = os.path.join(current_dir, "mycsp_analysis_1parcel.py")  

# Starting time
start_time = 0.0345

# Find all time folders and filter by start_time
time_folders = sorted(
    [f for f in os.listdir(folder_dir) 
     if os.path.isdir(os.path.join(folder_dir, f)) and float(f) >= start_time]
)

print(f"Found {len(time_folders)} time folders: {time_folders}")

# Loop over each folder and call the original script
for time_name in time_folders:
    file_path = os.path.join(folder_dir, time_name, "airCloud.dat")
    if not os.path.exists(file_path):
        print(f"[{time_name}] Skipping (no airCloud.dat)")
        continue

    print(f"\n[{time_name}] Running compute script for {file_path}")

    # Call the existing script with the path as argument
    subprocess.run(["python3", script_path, file_path, time_name])

