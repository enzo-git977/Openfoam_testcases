import cantera as ct
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, '/home/maffeise/OpenFOAM/maffeise-12/run/testcase_h2_Lean/PyCSP') # make sure we're working with the fct in this directory
import PyCSP.Functions as csp

# === Output directories ===
os.makedirs('data_Lean/', exist_ok=True)

# === Set up Cantera gas ===
gas = csp.CanteraCSP('mech/New_h_mech.yaml')

T = 1000
P = 2*ct.one_atm
gas.TP = T, P
gas.constP = True

# Set equivalence ratio
phi = 0.45
gas.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')

# Reactor setup
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
states = ct.SolutionArray(gas, extra=['t'])

# === CSP data containers ===
evals = []
Revec = []
Levec = []
fvec = []


print("\nMole fractions (ϕ = 0.45):")
for sp, X in zip(gas.species_names, gas.X):
    if X > 1e-10:
        print(f"{sp:10s}: {X:.6f}")
"""

# === Time integration (match OF timestep)===
t_end = 1e-3
dt = 1e-6 # Delta t of OF 
time = 0.0
while time <= t_end:
    #sim.step()
    sim.advance(time)
    states.append(r.thermo.state, t=time)
    print('%10.3e %10.3f %10.3f %14.6e' % (time, r.T, r.thermo.P, r.thermo.u))
    lam, R, L, f = gas.get_kernel()
    evals.append(lam)
    Revec.append(R)
    Levec.append(L)
    fvec.append(f)
    time+=dt
            
       
# === Save CSP arrays ===
np.savez_compressed(
    'data_Lean/csp_data.npz',
    evals=np.array(evals),
    Revec=np.array(Revec),
    Levec=np.array(Levec),
    fvec=np.array(fvec)
)
print("CSP arrays saved to csp_data.npz")

# === Save thermochemical state to CSV with '#' header ===
csv_path = 'data_Lean/cantera_states.csv'

# Create DataFrame
df_states = pd.DataFrame(states.T, columns=['T'])
df_states['time'] = states.t
for i, sp in enumerate(gas.species_names):
    df_states[sp] = states.Y[:, i]

# Reorder columns to put time first
cols = ['time', 'T'] + gas.species_names
df_states = df_states[cols]

# Write CSV with '#' before header
with open(csv_path, 'w') as f:
    f.write('#' + ','.join(df_states.columns) + '\n')
    df_states.to_csv(f, index=False, header=False)

print(f"Thermochemical state saved to {csv_path}")
"""
