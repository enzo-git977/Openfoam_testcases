print("Starting execution")
import os
import pandas as pd

# === Setup directory of time folders===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  	      # Get the current script's directory h2
Fields_DIR = os.path.join(CURRENT_DIR, "Openfoam_solution","postProcessing","probes","0") # directory of the fields file

# === List all field files in the directory ===
field_files = [f for f in os.listdir(Fields_DIR) if os.path.isfile(os.path.join(Fields_DIR, f))]

# === Initialize dictionary to hold DataFrames ===
field_dfs = []

# === Read each field file ===
for filename in field_files:
    field_name = filename  # filename is the field name (e.g., "T", "H2", etc.)
    file_path = os.path.join(Fields_DIR, filename)
    
    try:
        # Skip the first two comment lines 
        df = pd.read_csv(file_path, sep='\s+', header=None, skiprows=2, names=['time', field_name])
        field_dfs.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# === Merge all DataFrames on 'time' ===
if field_dfs:
    merged_df = field_dfs[0]
    for df in field_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='time', how='outer')

    # Sort by time
    merged_df.sort_values(by='time', inplace=True)

    # Write to CSV with header line starting with #
    output_file = "data_Foam_Lean.csv"
    with open(output_file, 'w') as f:
        f.write("# " + ",".join(merged_df.columns) + "\n")
        merged_df.to_csv(f, index=False, header=False)

    print(f"Data extracted to {output_file}")
else:
    print("No valid field files found.")

