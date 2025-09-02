print("starting execution")

#import numpy as np 
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import os
import sys


# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 	      	
    	
# Construct path  
time = "0.0025"
file_path = os.path.join(CURRENT_DIR, "postProcessing", "axialLine", time, "line1.xy")
output_dir = 'postProcessing/figures/'
os.makedirs(output_dir, exist_ok=True)

# Load OpenFOAM data
foam_data = pd.read_csv(file_path, comment='#', header=None,delim_whitespace=True)
foam_data.columns = ['x','T','H2','H','O','O2','OH','H2O','HO2','H2O2','AR', 'N2']

# Time and temperature
x_foam = foam_data['x']
T_foam = foam_data['T']

# Species mass fractions from OpenFOAM
species_compare = ['H', 'O2', 'N2']
Y_foam = {sp: foam_data[sp] for sp in species_compare}
print("Loaded thermo data from Foam")

# plot
# Plot Temperature
plt.subplot(2, 2, 1)
plt.plot(x_foam*1000, T_foam)
plt.xlabel('Position (mm)')
plt.ylabel('Temperature (K)')

# Plot H
plt.subplot(2, 2, 2)
plt.plot(x_foam*1000, Y_foam['H'])
plt.xlabel('Position (mm)')
plt.ylabel('H Mass Fraction')

# Plot O2
plt.subplot(2, 2, 3)
plt.plot(x_foam*1000, Y_foam['O2'])
plt.xlabel('Position (mm)')
plt.ylabel('O2 Mass Fraction')

# Plot N2
plt.subplot(2, 2, 4)
plt.plot(x_foam*1000, Y_foam['N2'])
plt.xlabel('Position (mm)')
plt.ylabel('N2 Mass Fraction')


plt.title("OpenFoam simulation")
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_{time}.png', dpi=500)
plt.show()
print(f"{output_dir}/plot_{time}.png")
