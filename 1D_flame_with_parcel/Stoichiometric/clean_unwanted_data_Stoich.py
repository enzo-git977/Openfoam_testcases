print("Starting execution...")
import os
import shutil

# === Paths =======================================================================================================
current_dir = os.getcwd()  # Gets current working directory
time_folders_path = os.path.join(current_dir, "Openfoam_solution-stoich-coarse", "postProcessing", "getEulerianFields")

# === FIND TIME FOLDERS ===
time_folders = sorted(
    [f for f in os.listdir(time_folders_path)
     if os.path.isdir(os.path.join(time_folders_path, f)) and f.replace('.', '', 1).isdigit()],
    key=lambda x: float(x)
)

# === REMOVE FOLDERS WITH t < 0.01 ================================================================================
SOI = 0.02
counter= 0
for folder in time_folders:
    folder_path = os.path.join(time_folders_path, folder)
    try:
        t = float(folder)
        if t < SOI:
            counter+=1
            shutil.rmtree(folder_path)
    except ValueError:
        print(f"Skipping non-numeric folder: {folder}")
        continue

print("Cleanup completed.")
print(f" Successfully removed {counter} empty files")

