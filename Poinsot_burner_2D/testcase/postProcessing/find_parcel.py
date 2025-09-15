import os
import sys

# === Config ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(CURRENT_DIR, "getEulerianFields")

# === Lister les dossiers de temps triés numériquement, sauf "0" ===
time_folders = sorted(
    [f for f in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, f))
     and f != "0"
     and f.replace('.', '', 1).isdigit()],
    key=lambda x: float(x)
)

# === Trouver le premier fichier airCloud.dat ayant des données ===
first_valid_time = None

for time_str in time_folders:
    file_path = os.path.join(base_dir, time_str, "airCloud.dat")

    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Garder les lignes qui ne sont pas des commentaires ou vides
            data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
            if data_lines:
                first_valid_time = time_str
                break

if first_valid_time:
    print(f" Premier dossier avec données dans airCloud.dat : {first_valid_time}")
    
else:
    print(" Aucun fichier airCloud.dat contenant des données trouvées.")
    sys.exit(1)

