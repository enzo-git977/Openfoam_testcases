print("Starting execution")
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os 

# Create the directories 
os.makedirs("flameProfiles", exist_ok=True)
#os.makedirs("data", exist_ok=True)
os.makedirs("data_new_mech", exist_ok=True)
os.makedirs("flameThicknessExp", exist_ok=True)
os.makedirs("flameSpeedExp", exist_ok=True)

# Parametri
phi = 0.45
pressure = ct.one_atm
T_in = 300.0
width = 0.05  # dominio [m]
grid=np.linspace(0,width,961)

results = []

gas = ct.Solution('mech/New_h_mech.yaml')
gas.transport_model = 'multicomponent'
gas.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')
gas.TP = T_in, pressure
print("solving flame...")
try:
    #f = ct.FreeFlame(gas, width=width)
    #f.set_refine_criteria(ratio=3, slope=0.03, curve=0.03, prune=0.001) # original
    #f.set_refine_criteria(ratio=2.0, slope=0.03, curve=0.03, prune=0.001)
    #f.solve(loglevel=0, auto=True)
    f = ct.FreeFlame(gas, grid=grid)
    f.soret_enabled = False
    f.solve(loglevel=0, auto=False)
    
    # Flame thickness
    T = f.T
    x = f.grid
    dTdx = (T[2:] - T[:-2]) / (x[2:] - x[:-2])
    indMax = np.argmax(dTdx)
    delta_f = (T[-1] - T[0]) / dTdx[indMax]
    delta_f_um = delta_f * 1e6  # µm
    
    # Laminar flame speed
    S_L = f.velocity[0]*100  # [cm/s]
    results.append((phi, delta_f_um, S_L))
    print(f"φ = {phi:.2f} → δ = {delta_f_um:.1f} µm, S_L = {S_L:.2f} cm/s")
    
except Exception as e:
    print(f"φ = {phi:.2f} → errore: {e}")
    
    
print("First point of domain :")
species_names = gas.species_names
Y_last = f.Y[:, 0]
for name, y in zip(species_names, Y_last):
    print(f"{name}: {y:.5e}")
print('T:',T[0])
#print('total:',0.028522387+0.745122605+0.226354006+1e-6)

print("================================")

print("last point of domain :")
species_names = gas.species_names
Y_last = f.Y[:, -1]
for name, y in zip(species_names, Y_last):
    print(f"{name}: {y:.5e}")
print('T:',T[-1])

print("================================")
"""
print("plotting Temperature")
# Plot temperature profile
plt.figure(figsize=(8, 5))
plt.plot(f.grid * 1000, f.T, label='Temperature', color='red')
plt.xlabel('Position [mm]')
plt.ylabel('Temperature [K]')
plt.title('Temperature Profile')
plt.grid(True)
plt.tight_layout()
plt.savefig("flameProfiles/temperature_profile.png", dpi=300)
plt.show()

print("plotting density")
# Plot density profile
plt.figure(figsize=(8, 5))
plt.plot(f.grid * 1000, f.density, label='density', color='blue')
plt.xlabel('Position [mm]')
plt.ylabel(' [kg/m3]')
plt.title('Density Profile')
plt.grid(True)
plt.tight_layout()
plt.savefig("flameProfiles/Density_profile.png", dpi=300)
plt.show()

print("plotting Species Mass fraction")
# Plot species profiles
species_to_plot = ['H2', 'O2', 'H2O', 'N', 'NO','H2O2']
num_species = len(species_to_plot)

fig, axs = plt.subplots(num_species, 1, figsize=(8, 2 * num_species), sharex=True)

for i, sp in enumerate(species_to_plot):
    axs[i].plot(f.grid * 1000, f.Y[gas.species_index(sp), :], label=sp, color='C' + str(i))
    axs[i].set_ylabel('Y_' + sp)
    axs[i].legend(loc='best')
    axs[i].grid(True)

axs[-1].set_xlabel('Position [mm]')
fig.suptitle('Species Mass Fraction Profiles', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Save and show
plt.savefig("flameProfiles/species_profiles_subplots.png", dpi=300)
plt.show()
"""
# Salvataggio dati: φ, flame thickness (µm), flame speed (m/s)
np.savetxt("flameThicknessExp/laminarFlamePropertiesCantera.xy",
           results, delimiter=',', comments='')

np.savetxt("flameSpeedExp/laminarFlamePropertiesCantera.xy",
           results, delimiter=',', comments='')

print("\nDati salvati in 'flameThicknessExp/laminarFlamePropertiesCantera.xy'")


# Select species to save
species_to_save = ['H', 'O', 'H2', 'O2', 'OH', 'H2O', 'N2', 'N', 'NO', 'HO2', 'H2O2', 'AR', 'OH*', 'HE']
species_indices = [gas.species_index(sp) for sp in species_to_save]

# Prepare data array
position_mm = f.grid * 1000  # [mm]
temperature_K = f.T
mass_fractions = np.vstack([f.Y[i, :] for i in species_indices])

# Stack columns: Position, Temp, Mass Fractions
data_to_save = np.column_stack([position_mm, temperature_K, mass_fractions.T])

# Header with hash
header = '# Position_mm,Temperature_K,' + ','.join(species_to_save)

# Save to CSV
np.savetxt(f"data_new_mech/cantera_profile.csv", data_to_save, delimiter=',', header=header, comments='')

print("Cantera profile saved to 'flameProfiles/cantera_profile.csv'")
