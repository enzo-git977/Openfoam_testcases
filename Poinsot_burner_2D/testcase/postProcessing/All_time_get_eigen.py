"""
Wrapper script to run `Get_eigen_vtk.py`
for all time folders in `cuttingPlanes`.
"""

import os
import subprocess

# Paths
current_dir = os.getcwd()
cutting_dir = os.path.join(current_dir, "cuttingPlanes")
script_path = os.path.join(current_dir, "Get_eigen_vtk.py")  

# Find all time folders
time_folders = sorted(
    [f for f in os.listdir(cutting_dir) if os.path.isdir(os.path.join(cutting_dir, f))]
)

print(f"Found {len(time_folders)} time folders: {time_folders}")

# Loop over each folder and call the original script
for time_name in time_folders:
    vtk_path = os.path.join(cutting_dir, time_name, "zMid.vtk")
    if not os.path.exists(vtk_path):
        print(f"[{time_name}] Skipping (no zMid.vtk)")
        continue

    print(f"\n[{time_name}] Running compute script for {vtk_path}")

    # Call the existing script with the path as argument
    subprocess.run(["python3", script_path, vtk_path, time_name])

