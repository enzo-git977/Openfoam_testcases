"""
Wrapper script to run `Plotting_vtk.py`
for all time folders in `cuttingPlanes` starting from a specific time.
"""

import os
import subprocess

# Paths
current_dir = os.getcwd()
cutting_dir = os.path.join(current_dir, "cuttingPlanes")
script_path = os.path.join(current_dir, "Plotting_vtk.py")  

# Starting time
start_time = 0.0345

# Find all time folders and filter by start_time
time_folders = sorted(
    [f for f in os.listdir(cutting_dir) 
     if os.path.isdir(os.path.join(cutting_dir, f)) and float(f) >= start_time]
)

print(f"Found {len(time_folders)} time folders starting from {start_time}: {time_folders}")

# Loop over each folder and call the plotting script
for time_name in time_folders:
    vtk_path = os.path.join(cutting_dir, time_name, "zMid.vtk")
    if not os.path.exists(vtk_path):
        print(f"[{time_name}] Skipping (no zMid.vtk)")
        continue

    print(f"\n[{time_name}] Running Plotting_vtk.py for {vtk_path}")
    subprocess.run(["python3", script_path, vtk_path, time_name])

